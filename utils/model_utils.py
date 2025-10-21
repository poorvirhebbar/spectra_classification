
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np

class CNN1D(nn.Module):
    """
    Improved 1D CNN with deeper architecture for better feature extraction.
    Uses 3 conv layers, wider channels, and dropout for regularization.
    """
    def __init__(self, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            # Layer 1: 1 -> 64 channels
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            
            # Layer 2: 64 -> 128 channels
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
            
            # Layer 3: 128 -> 256 channels
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
            
            nn.AdaptiveAvgPool1d(1),  # -> (N, 256, 1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),             # -> (N, 256)
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.6),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
    
    def extract_features(self, x):
        """
        Extract features from the penultimate layer (before final classification).
        Useful for visualization and analysis.
        
        Args:
            x: Input tensor (N, 1, L)
        
        Returns:
            features: (N, 128) feature tensor from second-to-last layer
        """
        x = self.features(x)  # (N, 256, 1) after adaptive pooling
        x = self.classifier[0](x)  # Flatten -> (N, 256)
        x = self.classifier[1](x)  # Dropout
        x = self.classifier[2](x)  # Linear(256, 128)
        x = self.classifier[3](x)  # ReLU
        # Don't apply final dropout and linear layer
        return x


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Focuses training on hard examples by down-weighting easy examples.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Class weights tensor (optional)
        gamma: Focusing parameter (default: 2.0). Higher gamma = more focus on hard examples
        reduction: 'mean' or 'sum'
    """
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (N, C) logits from model
            targets: (N,) class indices
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        
        # Get probabilities
        p = F.softmax(inputs, dim=1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)  # probability of true class
        
        # Compute focal term: (1 - p_t)^gamma
        focal_term = (1 - p_t) ** self.gamma
        
        # Focal loss
        loss = focal_term * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss




def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        total_items += xb.size(0)
    return total_loss / max(total_items, 1)

@torch.no_grad()
def evaluate(model, loader, criterion, device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    correct = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        total_items += xb.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == yb).sum().item()
    acc = correct / max(total_items, 1)
    return total_loss / max(total_items, 1), acc


def train_one_epoch_semisupervised(model, labelled_loader, unlabelled_loader, 
                                   criterion, optimizer, device, alpha, 
                                   conf_thresholds):
    """
    Train for one epoch with semi-supervised learning (pseudo-labeling).
    
    Args:
        model: CNN model
        labelled_loader: DataLoader for labelled data
        unlabelled_loader: DataLoader for unlabelled data
        criterion: Loss function (e.g., FocalLoss)
        optimizer: Optimizer
        device: torch device
        alpha: Weight for unsupervised loss (0 to 1)
        conf_thresholds: Dict mapping class_id -> confidence threshold
    
    Returns:
        Dictionary with training statistics:
            - 'loss_sup': Average supervised loss
            - 'loss_unsup': Average unsupervised loss
            - 'total_loss': Average total loss
            - 'pseudo_accept_rate': Fraction of pseudo-labels accepted
            - 'pseudo_accuracy': Accuracy of pseudo-labels (if possible to measure)
    """
    import torch.nn.functional as F
    
    model.train()
    unlabelled_iter = iter(unlabelled_loader)
    
    total_loss_supervised = 0.0
    total_loss_unsupervised = 0.0
    n_pseudo_accepted = 0
    n_pseudo_total = 0
    n_batches = 0
    
    # Track pseudo-label stats per class
    pseudo_stats = {
        'class_counts': {},
        'class_confidences': {},
    }
    
    for X_lab, y_lab in labelled_loader:
        X_lab, y_lab = X_lab.to(device), y_lab.to(device)
        
        # Supervised loss
        logits_lab = model(X_lab)
        loss_sup = criterion(logits_lab, y_lab)
        
        # Unsupervised loss (if alpha > 0)
        loss_unsup = torch.tensor(0.0, device=device)
        if alpha > 0:
            try:
                X_unlab, _ = next(unlabelled_iter)
            except StopIteration:
                # Restart unlabelled iterator if exhausted
                unlabelled_iter = iter(unlabelled_loader)
                X_unlab, _ = next(unlabelled_iter)
            
            X_unlab = X_unlab.to(device)
            
            # Generate pseudo-labels
            with torch.no_grad():
                logits_unlab = model(X_unlab)
                probs = F.softmax(logits_unlab, dim=1)
                confidence, pseudo_labels = probs.max(dim=1)
                
                # Filter by class-specific thresholds
                mask = torch.zeros(len(pseudo_labels), dtype=torch.bool, device=device)
                for class_id, threshold in conf_thresholds.items():
                    class_mask = (pseudo_labels == class_id) & (confidence >= threshold)
                    mask |= class_mask
            
            # Compute unsupervised loss only for confident samples
            if mask.sum() > 0:
                logits_unlab_filtered = model(X_unlab[mask])
                loss_unsup = criterion(logits_unlab_filtered, pseudo_labels[mask])
                n_pseudo_accepted += mask.sum().item()
                
                # Track pseudo-label statistics
                for class_id in conf_thresholds.keys():
                    class_pseudo_mask = (pseudo_labels[mask] == class_id)
                    if class_pseudo_mask.sum() > 0:
                        count = class_pseudo_mask.sum().item()
                        pseudo_stats['class_counts'][class_id] = \
                            pseudo_stats['class_counts'].get(class_id, 0) + count
            
            n_pseudo_total += len(X_unlab)
        
        # Combined loss
        total_loss = loss_sup + alpha * loss_unsup
        
        # Backprop
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        total_loss_supervised += loss_sup.item()
        if isinstance(loss_unsup, torch.Tensor):
            total_loss_unsupervised += loss_unsup.item()
        
        n_batches += 1
    
    # Compute averages
    avg_loss_sup = total_loss_supervised / max(n_batches, 1)
    avg_loss_unsup = total_loss_unsupervised / max(n_batches, 1)
    pseudo_accept_rate = n_pseudo_accepted / max(n_pseudo_total, 1)
    
    return {
        'loss_sup': avg_loss_sup,
        'loss_unsup': avg_loss_unsup,
        'total_loss': avg_loss_sup + alpha * avg_loss_unsup,
        'pseudo_accept_rate': pseudo_accept_rate,
        'pseudo_class_distribution': pseudo_stats['class_counts'],
    }
