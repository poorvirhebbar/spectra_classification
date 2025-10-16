"""
Real-time latent space visualization during training.
Shows how feature representations evolve epoch-by-epoch.

Saves visualizations to: training_feature_visualizations/run_XXX/epoch_YYY.png
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import json
from datetime import datetime
from sklearn.manifold import TSNE

# Try to import UMAP, fall back to t-SNE
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    print("UMAP not installed. Using t-SNE instead.")
    print("For better results: pip install umap-learn")
    HAS_UMAP = False


CLASS_NAMES = {0: "AGN", 1: "HM/LM/YSO", 2: "CV", 3: "NS/HMXB/LMXB/NS_BIN"}
CLASS_COLORS = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c', 3: '#d62728'}


class TrainingVisualizer:
    """
    Visualize latent space evolution during training.
    """
    
    def __init__(self, output_dir="training_feature_visualizations", 
                 num_classes=4, run_name=None, method='tsne'):
        """
        Initialize training visualizer.
        
        Args:
            output_dir: Base directory for saving visualizations
            num_classes: Number of classes
            run_name: Optional custom run name
            method: 'umap' or 'tsne' (tsne is faster for during-training viz)
        """
        self.base_dir = Path(output_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Auto-increment run number
        self.run_dir = self._create_run_directory(run_name)
        self.num_classes = num_classes
        self.method = method if (method == 'umap' and HAS_UMAP) else 'tsne'
        
        # Save run metadata
        self._save_metadata()
        
        print(f"\n📊 Training Visualizer initialized")
        print(f"   Run directory: {self.run_dir}")
        print(f"   Method: {self.method.upper()}")
        
    def _create_run_directory(self, run_name=None):
        """Create a new run directory with auto-incremented number."""
        if run_name:
            run_dir = self.base_dir / run_name
            run_dir.mkdir(exist_ok=True)
            return run_dir
        
        # Find existing run directories
        existing_runs = [d for d in self.base_dir.iterdir() 
                        if d.is_dir() and d.name.startswith('run_')]
        
        if existing_runs:
            # Extract run numbers
            run_numbers = []
            for run in existing_runs:
                try:
                    num = int(run.name.split('_')[1])
                    run_numbers.append(num)
                except (IndexError, ValueError):
                    continue
            next_run = max(run_numbers) + 1 if run_numbers else 1
        else:
            next_run = 1
        
        run_dir = self.base_dir / f"run_{next_run:03d}"
        run_dir.mkdir(exist_ok=True)
        return run_dir
    
    def _save_metadata(self):
        """Save metadata about this training run."""
        metadata = {
            'run_directory': str(self.run_dir),
            'num_classes': self.num_classes,
            'method': self.method,
            'start_time': datetime.now().isoformat(),
            'has_umap': HAS_UMAP
        }
        
        with open(self.run_dir / 'run_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    @torch.no_grad()
    def extract_features(self, model, dataloader, device, max_samples=500):
        """
        Extract features from model for a subset of data.
        Uses stratified sampling to ensure balanced class representation.
        
        Args:
            model: The CNN model
            dataloader: DataLoader for the data
            device: torch device
            max_samples: Maximum samples to extract (for speed)
        
        Returns:
            features, labels, predictions (numpy arrays)
        """
        model.eval()
        
        # First pass: extract ALL features (we'll subsample later)
        all_features = []
        all_labels = []
        all_preds = []
        
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            
            # Use the extract_features method if available, else use forward
            if hasattr(model, 'extract_features'):
                features = model.extract_features(xb)
            else:
                # Fallback: extract from penultimate layer
                x = model.features(xb)
                x = model.classifier[0](x)  # Flatten
                features = model.classifier[2](x)  # After first linear layer
            
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            
            all_features.append(features.cpu().numpy())
            all_labels.append(yb.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
        
        # Concatenate all
        features = np.vstack(all_features)
        labels = np.concatenate(all_labels)
        predictions = np.concatenate(all_preds)
        
        # If we have more samples than max_samples, do stratified sampling
        if len(features) > max_samples:
            # Sample equally from each class
            unique_classes = np.unique(labels)
            samples_per_class = max(max_samples // len(unique_classes), 50)  # At least 50 per class
            
            selected_indices = []
            for class_id in unique_classes:
                class_indices = np.where(labels == class_id)[0]
                n_samples = min(samples_per_class, len(class_indices))
                
                # Random sample from this class
                if n_samples < len(class_indices):
                    selected = np.random.choice(class_indices, n_samples, replace=False)
                else:
                    selected = class_indices
                
                selected_indices.extend(selected)
            
            # Shuffle the selected indices
            np.random.shuffle(selected_indices)
            
            # Subsample
            features = features[selected_indices]
            labels = labels[selected_indices]
            predictions = predictions[selected_indices]
        
        return features, labels, predictions
    
    def reduce_dimensions(self, features, random_state=42):
        """Reduce features to 2D quickly."""
        if self.method == 'umap' and HAS_UMAP:
            reducer = UMAP(n_components=2, random_state=random_state,
                          n_neighbors=15, min_dist=0.1, metric='euclidean',
                          n_epochs=200)  # Fewer epochs for speed
        else:
            reducer = TSNE(n_components=2, random_state=random_state,
                          perplexity=min(30, len(features) // 4),
                          n_iter=500)  # Fewer iterations for speed
        
        reduced = reducer.fit_transform(features)
        return reduced
    
    def visualize_epoch(self, model, train_loader, val_loader, device,
                       epoch, train_acc, val_acc, train_loss, val_loss):
        """
        Create and save visualization for current epoch.
        
        Args:
            model: The model to visualize
            train_loader: Training data loader
            val_loader: Validation data loader
            device: torch device
            epoch: Current epoch number
            train_acc: Training accuracy
            val_acc: Validation accuracy
            train_loss: Training loss
            val_loss: Validation loss
        """
        print(f"\n   📸 Creating latent space visualization for epoch {epoch}...")
        
        # Extract features from both sets
        train_features, train_labels, train_preds = self.extract_features(
            model, train_loader, device, max_samples=500
        )
        val_features, val_labels, val_preds = self.extract_features(
            model, val_loader, device, max_samples=300
        )
        
        # Combine for joint embedding
        all_features = np.vstack([train_features, val_features])
        all_labels = np.concatenate([train_labels, val_labels])
        all_preds = np.concatenate([train_preds, val_preds])
        is_train = np.array([True] * len(train_features) + [False] * len(val_features))
        
        # Reduce dimensions
        reduced = self.reduce_dimensions(all_features, random_state=epoch)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Left: Training data
        ax = axes[0]
        train_mask = is_train
        for class_id in range(self.num_classes):
            mask = (all_labels == class_id) & train_mask
            if mask.sum() > 0:
                ax.scatter(reduced[mask, 0], reduced[mask, 1],
                          c=CLASS_COLORS[class_id], label=CLASS_NAMES[class_id],
                          alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel(f'{self.method.upper()} 1', fontsize=12)
        ax.set_ylabel(f'{self.method.upper()} 2', fontsize=12)
        ax.set_title(f'Training Set - Epoch {epoch}\n'
                    f'Acc: {train_acc:.2%} | Loss: {train_loss:.4f}',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', frameon=True, fontsize=10)
        ax.grid(alpha=0.3)
        
        # Right: Validation data with misclassifications
        ax = axes[1]
        val_mask = ~is_train
        correct = (all_labels == all_preds) & val_mask
        
        # Plot correct predictions
        for class_id in range(self.num_classes):
            mask = (all_labels == class_id) & val_mask & correct
            if mask.sum() > 0:
                ax.scatter(reduced[mask, 0], reduced[mask, 1],
                          c=CLASS_COLORS[class_id], label=f'{CLASS_NAMES[class_id]} ✓',
                          alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        # Plot misclassifications
        for class_id in range(self.num_classes):
            mask = (all_labels == class_id) & val_mask & (~correct)
            if mask.sum() > 0:
                ax.scatter(reduced[mask, 0], reduced[mask, 1],
                          c=CLASS_COLORS[class_id], label=f'{CLASS_NAMES[class_id]} ✗',
                          alpha=0.8, s=100, edgecolors='red', linewidth=2, marker='X')
        
        ax.set_xlabel(f'{self.method.upper()} 1', fontsize=12)
        ax.set_ylabel(f'{self.method.upper()} 2', fontsize=12)
        ax.set_title(f'Validation Set - Epoch {epoch}\n'
                    f'Acc: {val_acc:.2%} | Loss: {val_loss:.4f}',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', frameon=True, fontsize=9, ncol=2)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        save_path = self.run_dir / f"epoch_{epoch:03d}_acc{val_acc:.4f}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved to: {save_path.name}")
        
        # Update progress file
        self._update_progress(epoch, train_acc, val_acc, train_loss, val_loss)
    
    def _update_progress(self, epoch, train_acc, val_acc, train_loss, val_loss):
        """Update progress log file."""
        progress_file = self.run_dir / 'training_progress.txt'
        
        with open(progress_file, 'a') as f:
            if epoch == 1 or not progress_file.exists():
                f.write("Epoch | Train Acc | Val Acc | Train Loss | Val Loss\n")
                f.write("-" * 60 + "\n")
            
            f.write(f"{epoch:5d} | {train_acc:9.4f} | {val_acc:7.4f} | "
                   f"{train_loss:10.4f} | {val_loss:8.4f}\n")
    
    def create_animation_script(self):
        """
        Create a script to generate an animation from saved images.
        Requires imagemagick or ffmpeg.
        """
        script_path = self.run_dir / 'create_animation.sh'
        
        with open(script_path, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("# Create animation from training visualizations\n\n")
            f.write("# Using imagemagick (install: sudo apt-get install imagemagick)\n")
            f.write("convert -delay 30 -loop 0 epoch_*.png training_animation.gif\n\n")
            f.write("# Or using ffmpeg (install: sudo apt-get install ffmpeg)\n")
            f.write("# ffmpeg -framerate 2 -pattern_type glob -i 'epoch_*.png' \\\n")
            f.write("#        -c:v libx264 -pix_fmt yuv420p training_evolution.mp4\n")
        
        script_path.chmod(0o755)
        print(f"\n💫 Animation script created: {script_path}")
        print(f"   Run: cd {self.run_dir} && ./create_animation.sh")


def should_visualize(epoch, total_epochs, visualize_every=10):
    """
    Determine if we should visualize at this epoch.
    
    Args:
        epoch: Current epoch
        total_epochs: Total number of epochs
        visualize_every: Visualize every N epochs
    
    Returns:
        bool: Whether to visualize
    """
    # Always visualize: first, last, and every N epochs
    if epoch == 1 or epoch == total_epochs:
        return True
    
    if epoch % visualize_every == 0:
        return True
    
    # Also visualize at key milestones
    milestones = [5, 10, 25, 50, 75, 100]
    if epoch in milestones and epoch <= total_epochs:
        return True
    
    return False

