"""
Semi-supervised learning utilities for pseudo-labeling.
Includes alpha scheduling and confidence filtering.
"""

import numpy as np
import torch
import torch.nn.functional as F


def alpha_schedule(epoch, total_epochs, warmup_epochs=50, max_alpha=0.8):
    """
    Curriculum learning schedule for unsupervised loss weight.
    
    Strategy:
    - Epochs 0 to warmup_epochs: alpha = 0 (supervised only)
    - Epochs warmup_epochs to total_epochs: alpha ramps from 0 to max_alpha
    
    Args:
        epoch: Current epoch (1-indexed)
        total_epochs: Total number of training epochs
        warmup_epochs: Number of epochs for warm-up (supervised only)
        max_alpha: Maximum alpha value (default: 0.8)
    
    Returns:
        alpha: Scalar weight for unsupervised loss
    
    Example:
        With total_epochs=200, warmup=50, max_alpha=0.8:
        - Epoch 1-50:   alpha = 0.0
        - Epoch 100:    alpha = 0.4
        - Epoch 150:    alpha = 0.67
        - Epoch 200:    alpha = 0.8
    """
    if epoch <= warmup_epochs:
        return 0.0
    
    # Linear ramp-up from warmup to total
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return min(max_alpha, progress * max_alpha)


def alpha_schedule_exponential(epoch, total_epochs, warmup_epochs=50, max_alpha=0.8):
    """
    Exponential curriculum learning schedule (slower start, faster end).
    
    Args:
        epoch: Current epoch (1-indexed)
        total_epochs: Total number of training epochs
        warmup_epochs: Number of epochs for warm-up
        max_alpha: Maximum alpha value
    
    Returns:
        alpha: Scalar weight for unsupervised loss
    """
    if epoch <= warmup_epochs:
        return 0.0
    
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    # Exponential curve: 1 - exp(-5*x) gives smooth ramp from 0 to 1
    alpha = max_alpha * (1.0 - np.exp(-5 * progress))
    return min(max_alpha, alpha)


def alpha_schedule_immediate(epoch, total_epochs, warmup_epochs=10, max_alpha=0.8):
    """
    Immediate start schedule - begins ramping up from epoch 1.
    Useful when model learns quickly and reaches good accuracy early.
    
    Strategy:
    - Epochs 1-warmup_epochs: alpha ramps from 0.1 to 0.3 (gentle start)
    - Epochs warmup_epochs to total_epochs: alpha ramps from 0.3 to max_alpha
    
    Args:
        epoch: Current epoch (1-indexed)
        total_epochs: Total number of training epochs
        warmup_epochs: Small warmup period (default: 10)
        max_alpha: Maximum alpha value (default: 0.8)
    
    Returns:
        alpha: Scalar weight for unsupervised loss
    
    Example:
        With total_epochs=200, warmup=10, max_alpha=0.8:
        - Epoch 1:    alpha = 0.1
        - Epoch 10:   alpha = 0.3
        - Epoch 50:   alpha = 0.5
        - Epoch 100:  alpha = 0.65
        - Epoch 200:  alpha = 0.8
    """
    if epoch <= warmup_epochs:
        # Gentle ramp from 0.1 to 0.3 in first few epochs
        progress = epoch / warmup_epochs
        return 0.1 + 0.2 * progress
    else:
        # Ramp from 0.3 to max_alpha
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return 0.3 + (max_alpha - 0.3) * progress


def filter_by_confidence_classwise(probs, pseudo_labels, thresholds):
    """
    Filter pseudo-labels by class-specific confidence thresholds.
    
    Args:
        probs: Softmax probabilities (N, num_classes)
        pseudo_labels: Predicted class labels (N,)
        thresholds: Dict mapping class_id -> threshold
                    e.g., {0: 0.95, 1: 0.95, 2: 0.90, 3: 0.85}
    
    Returns:
        mask: Boolean tensor (N,) indicating which samples pass threshold
    """
    confidence, _ = probs.max(dim=1)  # Get max probability
    
    # Build mask: check if confidence > threshold for that class
    mask = torch.zeros(len(pseudo_labels), dtype=torch.bool, device=probs.device)
    
    for class_id, threshold in thresholds.items():
        class_mask = (pseudo_labels == class_id) & (confidence >= threshold)
        mask |= class_mask
    
    return mask


def get_confidence_thresholds(num_classes, mode="4class"):
    """
    Get default confidence thresholds for each class.
    
    Args:
        num_classes: Number of classes (2 or 4)
        mode: "2class" or "4class"
    
    Returns:
        thresholds: Dict mapping class_id -> confidence threshold
    """
    if num_classes == 2:
        # 2-class: Both classes well-balanced, use same threshold
        return {
            0: 0.90,  # AGN
            1: 0.90,  # HM/LM/YSO
        }
    else:  # 4-class
        # Class-specific thresholds based on difficulty/frequency
        return {
            0: 0.95,  # AGN (majority, high confidence)
            1: 0.95,  # HM/LM/YSO (majority, high confidence)
            2: 0.90,  # CV (minority, slightly lower)
            3: 0.85,  # NS (very minority, more lenient)
        }


def compute_pseudo_label_stats(probs, pseudo_labels, true_labels=None):
    """
    Compute statistics about pseudo-labels for monitoring.
    
    Args:
        probs: Softmax probabilities (N, num_classes)
        pseudo_labels: Predicted class labels (N,)
        true_labels: Optional true labels (N,) if available
    
    Returns:
        stats: Dictionary with pseudo-label statistics
    """
    confidence, _ = probs.max(dim=1)
    
    stats = {
        'mean_confidence': confidence.mean().item(),
        'median_confidence': confidence.median().item(),
        'min_confidence': confidence.min().item(),
        'max_confidence': confidence.max().item(),
        'class_distribution': {},
    }
    
    # Class distribution
    unique_labels, counts = torch.unique(pseudo_labels, return_counts=True)
    for label, count in zip(unique_labels.cpu().numpy(), counts.cpu().numpy()):
        stats['class_distribution'][int(label)] = int(count)
    
    # If true labels available, compute accuracy
    if true_labels is not None:
        correct = (pseudo_labels == true_labels).sum().item()
        total = len(true_labels)
        stats['pseudo_accuracy'] = correct / total if total > 0 else 0.0
    
    return stats


def update_confidence_thresholds_adaptive(stats, current_thresholds, min_accept_rate=0.3):
    """
    Adaptively adjust confidence thresholds based on acceptance rate.
    
    If too few samples pass (< min_accept_rate), slightly lower thresholds.
    This is optional and can be enabled for more aggressive pseudo-labeling.
    
    Args:
        stats: Statistics from compute_pseudo_label_stats
        current_thresholds: Current threshold dict
        min_accept_rate: Minimum desired acceptance rate
    
    Returns:
        new_thresholds: Adjusted thresholds (or same if no adjustment needed)
    """
    # This is a placeholder for adaptive thresholding
    # For now, just return current thresholds (no adaptation)
    return current_thresholds


def create_mixed_dataset(X_labelled, y_labelled, X_unlabelled, 
                        sample_weights_labelled=None, unlabelled_ratio=1.0):
    """
    Create a mixed dataset with labelled and unlabelled samples.
    
    Args:
        X_labelled: Labelled features (N_lab, ...)
        y_labelled: Labelled targets (N_lab,)
        X_unlabelled: Unlabelled features (N_unlab, ...)
        sample_weights_labelled: Optional weights for labelled samples
        unlabelled_ratio: Ratio of unlabelled to labelled samples per batch
    
    Returns:
        Combined arrays with indicators for labelled/unlabelled
    """
    # This is a placeholder - actual mixing happens in training loop
    # Just return as-is for now
    return X_labelled, y_labelled, X_unlabelled


def save_pseudo_labels(pseudo_labels, confidences, source_ids, save_path):
    """
    Save pseudo-labels to file for inspection.
    
    Args:
        pseudo_labels: Predicted class labels
        confidences: Confidence scores
        source_ids: Source IDs for each sample
        save_path: Path to save JSON file
    """
    import json
    
    data = []
    for src_id, label, conf in zip(source_ids, pseudo_labels, confidences):
        data.append({
            'source_id': src_id,
            'pseudo_label': int(label),
            'confidence': float(conf)
        })
    
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved {len(data)} pseudo-labels to {save_path}")

