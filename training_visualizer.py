"""
Real-time latent space visualization during training.
Shows how feature representations evolve epoch-by-epoch.
Enhanced with GradCAM and activation visualization for 1D spectra.

Saves visualizations to: training_feature_visualizations/run_XXX/epoch_YYY.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch
import torch.nn.functional as F
from pathlib import Path
import json
from datetime import datetime
from sklearn.manifold import TSNE
from scipy.signal import find_peaks
import cv2

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


class GradCAM1D:
    """
    GradCAM implementation for 1D CNN models.
    Visualizes which parts of the input spectrum the model focuses on.
    """
    
    def __init__(self, model, target_layer_name):
        """
        Initialize GradCAM for 1D data.
        
        Args:
            model: The CNN model
            target_layer_name: Name of the target layer (e.g., 'features.2' for conv layer 3)
        """
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Find the target layer
        target_layer = None
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                target_layer = module
                break
        
        if target_layer is None:
            raise ValueError(f"Layer '{self.target_layer_name}' not found in model")
        
        # Register hooks
        self.hooks.append(target_layer.register_forward_hook(forward_hook))
        self.hooks.append(target_layer.register_backward_hook(backward_hook))
    
    def generate_cam(self, input_tensor, class_idx=None):
        """
        Generate GradCAM for the given input.
        
        Args:
            input_tensor: Input tensor (1, 1, L) or (N, 1, L)
            class_idx: Class index to generate CAM for. If None, uses predicted class.
        
        Returns:
            cam: GradCAM heatmap (L,) or (N, L)
        """
        self.model.eval()
        
        # Ensure input is on the same device as model
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        # Forward pass
        logits = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1)
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, class_idx.unsqueeze(1), 1.0)
        
        logits.backward(gradient=one_hot, retain_graph=True)
        
        # Generate CAM
        gradients = self.gradients  # (N, C, L)
        activations = self.activations  # (N, C, L)
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=2, keepdim=True)  # (N, C, 1)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1)  # (N, L)
        
        # Apply ReLU to get only positive contributions
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam - cam.min(dim=1, keepdim=True)[0]
        cam = cam / (cam.max(dim=1, keepdim=True)[0] + 1e-8)
        
        return cam.cpu().numpy()
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def detect_emission_lines(spectrum, prominence=0.1, distance=5):
    """
    Detect emission lines (local maxima) in a spectrum.
    
    Args:
        spectrum: 1D array of spectral counts
        prominence: Minimum prominence for peak detection
        distance: Minimum distance between peaks
    
    Returns:
        peaks: Indices of detected peaks
        properties: Peak properties
    """
    peaks, properties = find_peaks(spectrum, prominence=prominence, distance=distance)
    return peaks, properties


class TrainingVisualizer:
    """
    Visualize latent space evolution during training.
    Enhanced with GradCAM and activation visualization for 1D spectra.
    """
    
    def __init__(self, output_dir="training_feature_visualizations", 
                 num_classes=4, run_name=None, method='tsne', 
                 enable_gradcam=True, gradcam_layer='features.2',
                 train_metadata=None, val_metadata=None):
        """
        Initialize training visualizer.
        
        Args:
            output_dir: Base directory for saving visualizations
            num_classes: Number of classes
            run_name: Optional custom run name
            method: 'umap' or 'tsne' (tsne is faster for during-training viz)
            enable_gradcam: Whether to enable GradCAM visualization
            gradcam_layer: Target layer for GradCAM (default: 'features.2' for conv layer 3)
            train_metadata: Dict with 'src_ids' and 'counts' for training data
            val_metadata: Dict with 'src_ids' and 'counts' for validation data
        """
        self.base_dir = Path(output_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Auto-increment run number
        self.run_dir = self._create_run_directory(run_name)
        self.num_classes = num_classes
        self.method = method if (method == 'umap' and HAS_UMAP) else 'tsne'
        self.enable_gradcam = enable_gradcam
        self.gradcam_layer = gradcam_layer
        
        # Store metadata for filtering and display
        self.train_metadata = train_metadata or {}
        self.val_metadata = val_metadata or {}
        
        # Save run metadata
        self._save_metadata()
        
        print(f"\n📊 Training Visualizer initialized")
        print(f"   Run directory: {self.run_dir}")
        print(f"   Method: {self.method.upper()}")
        if self.enable_gradcam:
            print(f"   GradCAM enabled on layer: {self.gradcam_layer}")
        
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
            'enable_gradcam': self.enable_gradcam,
            'gradcam_layer': self.gradcam_layer,
            'start_time': datetime.now().isoformat(),
            'has_umap': HAS_UMAP
        }
        
        with open(self.run_dir / 'run_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    @torch.no_grad()
    def extract_features(self, model, dataloader, device, max_samples=500, metadata=None, min_count=500):
        """
        Extract features from model for a subset of data.
        Uses stratified sampling to ensure balanced class representation.
        
        Args:
            model: The CNN model
            dataloader: DataLoader for the data
            device: torch device
            max_samples: Maximum samples to extract (for speed)
            metadata: Dict with 'src_ids' and 'counts' arrays
            min_count: Minimum count threshold for filtering samples (default: 500)
        
        Returns:
            features, labels, predictions, src_ids (numpy arrays)
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
        
        # Get source IDs and counts from metadata
        src_ids = metadata.get('src_ids', np.array([f"sample_{i}" for i in range(len(features))]))
        counts = metadata.get('counts', np.ones(len(features)) * min_count)  # Default to passing filter
        
        # Filter by count > min_count (only if counts array matches features size)
        if len(counts) == len(features):
            count_mask = counts > min_count
            features = features[count_mask]
            labels = labels[count_mask]
            predictions = predictions[count_mask]
            src_ids = src_ids[count_mask]
        else:
            # Size mismatch (e.g., after data augmentation) - skip count filtering
            # Use first len(features) source IDs or generate defaults
            if len(src_ids) < len(features):
                src_ids = np.array([f"sample_{i}" for i in range(len(features))])
        
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
            src_ids = src_ids[selected_indices]
        
        return features, labels, predictions, src_ids
    
    @torch.no_grad()
    def extract_sample_spectra(self, model, dataloader, device, metadata=None, num_samples=6, min_count=500, skip_first=2):
        """
        Extract sample spectra and their predictions for GradCAM visualization.
        
        Args:
            model: The CNN model
            dataloader: DataLoader for the data
            device: torch device
            metadata: Dict with 'src_ids' and 'counts' arrays
            num_samples: Number of samples to extract per class
            min_count: Minimum count threshold for filtering samples
            skip_first: Number of samples to skip per class before selecting (default: 2)
        
        Returns:
            spectra, labels, predictions, src_ids (numpy arrays)
        """
        model.eval()
        
        all_spectra = []
        all_labels = []
        all_preds = []
        all_indices = []  # Track which sample indices we're using
        
        # Collect samples from each class (with skip_first offset)
        class_counts = {i: 0 for i in range(self.num_classes)}
        class_skipped = {i: 0 for i in range(self.num_classes)}  # Track how many we've skipped per class
        sample_idx = 0
        
        # Get metadata
        src_ids = metadata.get('src_ids', np.array([f"sample_{i}" for i in range(1000)])) if metadata else None
        counts = metadata.get('counts', np.ones(1000) * min_count) if metadata else None
        
        for xb, yb in dataloader:
            if all(count >= num_samples for count in class_counts.values()):
                break
                
            xb, yb = xb.to(device), yb.to(device)
            
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            
            # Select samples from classes we still need (after skipping first N)
            for i in range(xb.size(0)):
                true_class = yb[i].item()
                
                # Check if sample passes count filter
                if counts is None or sample_idx >= len(counts) or counts[sample_idx] > min_count:
                    # Skip the first N samples from this class
                    if class_skipped[true_class] < skip_first:
                        class_skipped[true_class] += 1
                    # Now collect samples after skipping
                    elif class_counts[true_class] < num_samples:
                        all_spectra.append(xb[i].cpu().numpy())
                        all_labels.append(yb[i].cpu().numpy())
                        all_preds.append(preds[i].cpu().numpy())
                        all_indices.append(sample_idx)
                        class_counts[true_class] += 1
                
                sample_idx += 1
        
        # Get source IDs for selected samples
        if src_ids is not None and len(all_indices) > 0:
            # Only use indices that are within bounds
            valid_indices = [idx for idx in all_indices if idx < len(src_ids)]
            if len(valid_indices) == len(all_indices):
                selected_src_ids = src_ids[all_indices]
            else:
                # Some indices out of bounds (augmented data) - generate defaults
                selected_src_ids = np.array([
                    src_ids[idx] if idx < len(src_ids) else f"sample_{idx}"
                    for idx in all_indices
                ])
        else:
            selected_src_ids = np.array([f"sample_{i}" for i in range(len(all_spectra))])
        
        return np.array(all_spectra), np.array(all_labels), np.array(all_preds), selected_src_ids
    
    def reduce_dimensions(self, features, random_state=42):
        """Reduce features to 2D quickly."""
        if self.method == 'umap' and HAS_UMAP:
            reducer = UMAP(n_components=2, random_state=random_state,
                          n_neighbors=15, min_dist=0.1, metric='euclidean',
                          n_epochs=200)  # Fewer epochs for speed
        else:
            # Handle different scikit-learn versions
            try:
                # Newer versions use 'max_iter'
                reducer = TSNE(n_components=2, random_state=random_state,
                              perplexity=min(30, len(features) // 4),
                              max_iter=500)  # Fewer iterations for speed
            except TypeError:
                # Older versions use 'n_iter'
                reducer = TSNE(n_components=2, random_state=random_state,
                              perplexity=min(30, len(features) // 4),
                              n_iter=500)  # Fewer iterations for speed
        
        reduced = reducer.fit_transform(features)
        return reduced
    
    def visualize_epoch(self, model, train_loader, val_loader, device,
                       epoch, train_acc, val_acc, train_loss, val_loss):
        """
        Create and save visualization for current epoch.
        Enhanced with GradCAM and activation visualization.
        
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
        print(f"\n   📸 Creating enhanced visualization for epoch {epoch}...")
        
        # Extract features from both sets for latent space visualization
        train_features, train_labels, train_preds, train_src_ids = self.extract_features(
            model, train_loader, device, max_samples=500, metadata=self.train_metadata, min_count=500
        )
        val_features, val_labels, val_preds, val_src_ids = self.extract_features(
            model, val_loader, device, max_samples=300, metadata=self.val_metadata, min_count=500
        )
        
        # Combine for joint embedding
        all_features = np.vstack([train_features, val_features])
        all_labels = np.concatenate([train_labels, val_labels])
        all_preds = np.concatenate([train_preds, val_preds])
        all_src_ids = np.concatenate([train_src_ids, val_src_ids])
        is_train = np.array([True] * len(train_features) + [False] * len(val_features))
        
        # Reduce dimensions
        reduced = self.reduce_dimensions(all_features, random_state=epoch)
        
        # Create visualization with multiple subplots
        if self.enable_gradcam:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Epoch {epoch} - Training Progress & Receptive Fields', 
                        fontsize=16, fontweight='bold')
        else:
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))
            axes = axes.reshape(1, -1)  # Make it 2D for consistent indexing
        
        # Left: Training data latent space
        ax = axes[0, 0]
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
        
        # Middle: Validation data with misclassifications
        ax = axes[0, 1]
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
        
        # GradCAM visualizations (if enabled)
        if self.enable_gradcam:
            try:
                # Extract sample spectra for GradCAM (skip first 3 samples to avoid consistently hard cases)
                sample_spectra, sample_labels, sample_preds, sample_src_ids = self.extract_sample_spectra(
                    model, val_loader, device, metadata=self.val_metadata, num_samples=2, min_count=500, skip_first=3
                )
                
                if len(sample_spectra) > 0:
                    # Initialize GradCAM
                    gradcam = GradCAM1D(model, self.gradcam_layer)
                    
                    # Generate GradCAM for each sample
                    cams = gradcam.generate_cam(torch.tensor(sample_spectra))
                    
                    # Plot GradCAM visualizations (2 samples in bottom row)
                    for i in range(min(2, len(sample_spectra))):
                        ax = axes[1, i]  # Bottom row, columns 0 and 1
                        
                        spectrum = sample_spectra[i].squeeze()  # Remove channel dimension
                        cam = cams[i]
                        true_label = sample_labels[i]
                        pred_label = sample_preds[i]
                        src_id = sample_src_ids[i] if i < len(sample_src_ids) else f"sample_{i}"
                        
                        # Create linearly-spaced bins from 0.5 to 10 keV, displayed on log scale
                        bins = np.linspace(0.5, 10, len(spectrum))
                        
                        # Plot spectrum
                        ax.plot(bins, spectrum, 'b-', linewidth=1, alpha=0.7, label='Spectrum')
                        
                        # Overlay GradCAM heatmap with proper scaling
                        # Scale CAM to match spectrum range for better visualization
                        # Use a more aggressive scaling to make attention visible
                        cam_scaled = cam * spectrum.max() * 0.8  # Scale down slightly for better visibility
                        ax.fill_between(bins, 0, cam_scaled, 
                                      alpha=0.4, color='red', label='Attention')
                        
                        # Formatting
                        ax.set_xlabel('Energy [keV]', fontsize=10)
                        ax.set_ylabel('Normalized counts/keV', fontsize=10)
                        ax.set_xscale('log')

                        # Start exactly at 0.5 and end at 10, with equal (small) margins both sides
                        ax.set_xlim(0.5, 10)
                        ax.margins(x=0.02)  # symmetric 2% breathing room on both ends

                        # Make sure these ticks (and labels) always appear
                        ax.set_xticks([0.5, 1, 2, 5, 10])
                        ax.set_xticklabels(['0.5', '1', '2', '5', '10'])
                        ax.xaxis.set_minor_formatter(mticker.NullFormatter())

                        ax.set_title(f'Sample {i+1} [{src_id}]: True={CLASS_NAMES[true_label]}, '
                                    f'Pred={CLASS_NAMES[pred_label]}', fontsize=10)
                        ax.legend(fontsize=8)
                        ax.grid(alpha=0.3)

                        
                        # Highlight high attention regions (more selective)
                        high_attention = cam > 0.5  # Lower threshold for better visibility
                        if np.any(high_attention):
                            high_attention_indices = np.where(high_attention)[0]
                            if len(high_attention_indices) > 0:
                                ax.axvspan(bins[high_attention_indices[0]], 
                                          bins[high_attention_indices[-1]], 
                                          alpha=0.1, color='red', label='High Attention')
                    
                    # Clean up GradCAM hooks
                    gradcam.remove_hooks()
                    
            except Exception as e:
                print(f"   ⚠️  GradCAM visualization failed: {e}")
                # Fill empty subplots in bottom row
                for i in range(2):
                    axes[1, i].text(0.5, 0.5, 'GradCAM\nFailed', 
                                   ha='center', va='center', transform=axes[1, i].transAxes)
                    axes[1, i].set_title('GradCAM Error', fontsize=10)
        
        plt.tight_layout()
        
        # Save
        save_path = self.run_dir / f"epoch_{epoch:03d}_acc{val_acc:.4f}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved enhanced visualization to: {save_path.name}")
        
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
        Automatically detects and uses available tools (ffmpeg or ImageMagick).
        """
        script_path = self.run_dir / 'create_animation.sh'
        
        with open(script_path, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("# Create animation from training visualizations\n\n")
            f.write("# Try ffmpeg first (more commonly available)\n")
            f.write("if command -v ffmpeg &> /dev/null; then\n")
            f.write("    echo \"Creating MP4 animation with ffmpeg...\"\n")
            f.write("    ffmpeg -framerate 2 -pattern_type glob -i 'epoch_*.png' \\\n")
            f.write("           -c:v libx264 -pix_fmt yuv420p training_evolution.mp4\n")
            f.write("    echo \"✅ Animation saved as: training_evolution.mp4\"\n")
            f.write("elif command -v convert &> /dev/null; then\n")
            f.write("    echo \"Creating GIF animation with ImageMagick...\"\n")
            f.write("    convert -delay 30 -loop 0 epoch_*.png training_animation.gif\n")
            f.write("    echo \"✅ Animation saved as: training_animation.gif\"\n")
            f.write("else\n")
            f.write("    echo \"❌ Neither ffmpeg nor ImageMagick found!\"\n")
            f.write("    echo \"Install one of them:\"\n")
            f.write("    echo \"  - ffmpeg: brew install ffmpeg (macOS) or sudo apt-get install ffmpeg (Ubuntu)\"\n")
            f.write("    echo \"  - ImageMagick: brew install imagemagick (macOS) or sudo apt-get install imagemagick (Ubuntu)\"\n")
            f.write("fi\n")
        
        script_path.chmod(0o755)
        print(f"\n💫 Animation script created: {script_path}")
        print(f"   Run: cd {self.run_dir} && ./create_animation.sh")


def should_visualize(epoch, total_epochs, visualize_every=5):
    """
    Determine if we should visualize at this epoch.
    
    Args:
        epoch: Current epoch
        total_epochs: Total number of epochs
        visualize_every: Visualize every N epochs (default: 5 for GradCAM)
    
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

