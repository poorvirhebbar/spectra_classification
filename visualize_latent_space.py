"""
Visualize latent space representations from the trained CNN model.
Uses UMAP for dimensionality reduction to 2D/3D.

Usage:
    python visualize_latent_space.py --checkpoint checkpoints/best_*.pt --classes 4
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import os
import re
from sklearn.manifold import TSNE
from pathlib import Path

from utils.model_utils import CNN1D
from utils.data_utils import load_combined_data, filter_and_remap, make_tensors

# Try to import UMAP, fall back to t-SNE if not available
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    print("UMAP not installed. Using t-SNE instead.")
    print("For better results: pip install umap-learn")
    HAS_UMAP = False


CLASS_NAMES = {0: "AGN", 1: "HM/LM/YSO", 2: "CV", 3: "NS/HMXB/LMXB/NS_BIN"}
CLASS_COLORS = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c', 3: '#d62728'}  # Blue, Orange, Green, Red


def parse_checkpoint_name(checkpoint_path):
    """
    Parse checkpoint filename to extract metadata.
    
    Example: "checkpoints/best_4cls_2730ex_380bins_val0.8567.pt"
    Returns: {
        'type': 'best',
        'classes': '4cls',
        'samples': '2730ex',
        'bins': '380bins',
        'accuracy': '0.8567',
        'basename': 'best_4cls_2730ex_380bins_val0.8567'
    }
    """
    filename = Path(checkpoint_path).stem  # Remove .pt extension
    
    info = {'basename': filename}
    
    # Extract checkpoint type (best/last)
    if 'best' in filename:
        info['type'] = 'best'
    elif 'last' in filename:
        info['type'] = 'last'
    else:
        info['type'] = 'checkpoint'
    
    # Extract number of classes
    match = re.search(r'(\d+)cls', filename)
    if match:
        info['classes'] = match.group(1) + 'cls'
    
    # Extract number of samples
    match = re.search(r'(\d+)ex', filename)
    if match:
        info['samples'] = match.group(1) + 'ex'
    
    # Extract number of bins
    match = re.search(r'(\d+)bins', filename)
    if match:
        info['bins'] = match.group(1) + 'bins'
    
    # Extract validation accuracy
    match = re.search(r'val([\d.]+)', filename)
    if match:
        info['accuracy'] = match.group(1)
        info['accuracy_pct'] = f"{float(match.group(1)) * 100:.2f}%"
    
    return info


def create_output_directory(checkpoint_path):
    """
    Create output directory based on checkpoint name.
    
    Returns: Path to the output directory
    """
    info = parse_checkpoint_name(checkpoint_path)
    
    # Create main visualization directory
    vis_dir = Path("visualizations")
    vis_dir.mkdir(exist_ok=True)
    
    # Create subdirectory based on checkpoint
    # Format: visualizations/best_4cls_acc85.67/
    if 'accuracy_pct' in info:
        subdir_name = f"{info['type']}_{info.get('classes', 'unknown')}_acc{info['accuracy_pct']}"
    else:
        subdir_name = info['basename']
    
    output_dir = vis_dir / subdir_name
    output_dir.mkdir(exist_ok=True)
    
    return output_dir, info


def extract_features(model, dataloader, device):
    """
    Extract features from the penultimate layer of the model.
    
    Returns:
        features: (N, feature_dim) numpy array
        labels: (N,) numpy array
        predictions: (N,) numpy array
    """
    model.eval()
    all_features = []
    all_labels = []
    all_preds = []
    
    # Register hook to capture features before final classification
    features_hook = []
    def hook_fn(module, input, output):
        # Capture output of the second-to-last layer (before final linear)
        features_hook.append(output.detach().cpu())
    
    # Find the last linear layer before output
    # Architecture: features -> flatten -> dropout -> linear -> relu -> dropout -> linear(output)
    # We want features after the first linear layer
    hook_handle = model.classifier[2].register_forward_hook(hook_fn)  # After first Linear + ReLU
    
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            
            all_labels.append(yb.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            
    # Remove hook
    hook_handle.remove()
    
    # Concatenate all batches
    features = torch.cat(features_hook, dim=0).numpy()
    labels = np.concatenate(all_labels)
    predictions = np.concatenate(all_preds)
    
    return features, labels, predictions


def reduce_dimensions(features, method='umap', n_components=2, random_state=42):
    """
    Reduce features to 2D or 3D using UMAP or t-SNE.
    """
    if method == 'umap' and HAS_UMAP:
        print(f"Reducing to {n_components}D using UMAP...")
        reducer = UMAP(n_components=n_components, random_state=random_state, 
                       n_neighbors=15, min_dist=0.1, metric='euclidean')
    else:
        print(f"Reducing to {n_components}D using t-SNE...")
        reducer = TSNE(n_components=n_components, random_state=random_state, 
                       perplexity=30, n_iter=1000)
    
    reduced = reducer.fit_transform(features)
    return reduced


def plot_latent_space_2d(reduced, labels, predictions, save_path=None):
    """
    Create a 2D visualization of the latent space.
    Shows both true labels and misclassifications.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Color by true label
    ax = axes[0]
    for class_id in np.unique(labels):
        mask = labels == class_id
        ax.scatter(reduced[mask, 0], reduced[mask, 1], 
                  c=CLASS_COLORS[class_id], label=CLASS_NAMES[class_id],
                  alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('UMAP 1' if HAS_UMAP else 't-SNE 1', fontsize=12)
    ax.set_ylabel('UMAP 2' if HAS_UMAP else 't-SNE 2', fontsize=12)
    ax.set_title('Latent Space - Colored by True Label', fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, fontsize=10)
    ax.grid(alpha=0.3)
    
    # Plot 2: Show misclassifications
    ax = axes[1]
    correct = labels == predictions
    
    # Plot correct predictions with low opacity
    ax.scatter(reduced[correct, 0], reduced[correct, 1], 
              c='lightgray', alpha=0.3, s=30, label='Correct', edgecolors='none')
    
    # Plot misclassifications by true class
    for class_id in np.unique(labels):
        mask = (labels == class_id) & (~correct)
        if mask.sum() > 0:
            ax.scatter(reduced[mask, 0], reduced[mask, 1], 
                      c=CLASS_COLORS[class_id], label=f'{CLASS_NAMES[class_id]} (wrong)',
                      alpha=0.8, s=100, edgecolors='red', linewidth=2, marker='X')
    
    ax.set_xlabel('UMAP 1' if HAS_UMAP else 't-SNE 1', fontsize=12)
    ax.set_ylabel('UMAP 2' if HAS_UMAP else 't-SNE 2', fontsize=12)
    ax.set_title('Latent Space - Highlighting Misclassifications', fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved 2D visualization to {save_path}")
    
    plt.show()


def plot_class_separability(reduced, labels, save_path=None):
    """
    Create pairwise density plots to show class separability.
    """
    n_classes = len(np.unique(labels))
    fig, axes = plt.subplots(n_classes, n_classes, figsize=(12, 12))
    
    for i in range(n_classes):
        for j in range(n_classes):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: show density of single class
                mask = labels == i
                ax.hist(reduced[mask, 0], bins=30, alpha=0.7, color=CLASS_COLORS[i])
                ax.set_ylabel(CLASS_NAMES[i], fontsize=10, fontweight='bold')
            else:
                # Off-diagonal: show overlap between two classes
                mask_i = labels == i
                mask_j = labels == j
                ax.scatter(reduced[mask_j, 0], reduced[mask_j, 1], 
                          c=CLASS_COLORS[j], alpha=0.3, s=20)
                ax.scatter(reduced[mask_i, 0], reduced[mask_i, 1], 
                          c=CLASS_COLORS[i], alpha=0.3, s=20)
            
            ax.set_xticks([])
            ax.set_yticks([])
            
            if i == 0:
                ax.set_title(CLASS_NAMES[j], fontsize=10)
    
    plt.suptitle('Class Separability Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved separability plot to {save_path}")
    
    plt.show()


def print_statistics(labels, predictions):
    """
    Print statistics about the latent space.
    """
    correct = labels == predictions
    accuracy = correct.sum() / len(labels)
    
    print("\n" + "="*60)
    print("LATENT SPACE STATISTICS")
    print("="*60)
    print(f"Total samples: {len(labels)}")
    print(f"Overall accuracy: {accuracy:.4f}")
    print()
    
    print("Per-class performance:")
    for class_id in sorted(np.unique(labels)):
        mask = labels == class_id
        class_correct = (predictions[mask] == class_id).sum()
        class_total = mask.sum()
        class_acc = class_correct / class_total if class_total > 0 else 0
        
        print(f"  {CLASS_NAMES[class_id]:20s}: {class_correct:3d}/{class_total:3d} correct ({class_acc:.2%})")
    
    print("\nMost confused pairs:")
    confusion_pairs = []
    for true_class in sorted(np.unique(labels)):
        for pred_class in sorted(np.unique(predictions)):
            if true_class != pred_class:
                mask = (labels == true_class) & (predictions == pred_class)
                count = mask.sum()
                if count > 0:
                    confusion_pairs.append((count, true_class, pred_class))
    
    confusion_pairs.sort(reverse=True)
    for count, true_class, pred_class in confusion_pairs[:5]:
        print(f"  {CLASS_NAMES[true_class]:15s} → {CLASS_NAMES[pred_class]:15s}: {count:3d} errors")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Visualize latent space of trained model")
    parser.add_argument("--checkpoint", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--classes", choices=["2", "4"], default="4", help="Number of classes")
    parser.add_argument("--data", default="data/Brightpn_id_normspec_counts_label_onlylabelled.txt")
    parser.add_argument("--method", choices=["umap", "tsne"], default="umap", 
                       help="Dimensionality reduction method")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--val_split", type=float, default=0.2)
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Create output directory based on checkpoint name
    output_dir, checkpoint_info = create_output_directory(args.checkpoint)
    print(f"\n📁 Output directory: {output_dir}")
    print(f"   Checkpoint info: {checkpoint_info}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    num_classes = checkpoint['num_classes']
    
    # Load data
    print(f"Loading data from {args.data}...")
    X_all, y_all, src_ids = load_combined_data(args.data)
    
    # Filter based on class mode
    if args.classes == "2":
        mode = "12"
        mask = np.isin(y_all, [0, 1])
        X_filtered = X_all[mask]
        y_filtered, _, target_names = filter_and_remap(y_all, mode)
    else:
        X_filtered = X_all
        y_filtered = y_all
        _, _, target_names = filter_and_remap(y_all, "all4")
    
    print(f"Using {len(y_filtered)} samples across {num_classes} classes")
    
    # Split data (same split as training)
    Xtr, Xva, ytr, yva = train_test_split(X_filtered, y_filtered, test_size=args.val_split,
                                          random_state=42, stratify=y_filtered)
    
    # Normalize
    def norm_per_sample(A):
        mu = A.mean(axis=1, keepdims=True)
        sigma = A.std(axis=1, keepdims=True) + 1e-8
        return (A - mu) / sigma
    
    Xva = norm_per_sample(Xva)
    
    # Create tensors and dataloader
    Xva_t, yva_t = make_tensors(Xva, yva)
    val_loader = DataLoader(TensorDataset(Xva_t, yva_t), batch_size=64, shuffle=False)
    
    # Load model
    print("Loading model...")
    model = CNN1D(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    # Extract features
    print("Extracting features from validation set...")
    features, labels, predictions = extract_features(model, val_loader, device)
    print(f"Extracted features shape: {features.shape}")
    
    # Print statistics
    print_statistics(labels, predictions)
    
    # Reduce dimensions
    reduced_2d = reduce_dimensions(features, method=args.method, n_components=2)
    
    # Create meaningful output filenames
    method_name = 'umap' if HAS_UMAP and args.method == 'umap' else 'tsne'
    accuracy_str = checkpoint_info.get('accuracy_pct', 'unknown')
    classes_str = checkpoint_info.get('classes', f"{args.classes}cls")
    
    latent_space_file = output_dir / f"latent_space_2d_{method_name}.png"
    separability_file = output_dir / f"class_separability_{method_name}.png"
    
    # Visualize
    print("\nCreating visualizations...")
    plot_latent_space_2d(reduced_2d, labels, predictions, save_path=str(latent_space_file))
    plot_class_separability(reduced_2d, labels, save_path=str(separability_file))
    
    # Save summary statistics to text file
    summary_file = output_dir / f"statistics.txt"
    with open(summary_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("LATENT SPACE VISUALIZATION SUMMARY\n")
        f.write("="*60 + "\n")
        f.write(f"Checkpoint: {Path(args.checkpoint).name}\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"Method: {method_name.upper()}\n")
        f.write(f"Classes: {num_classes}\n")
        f.write(f"Validation accuracy: {accuracy_str}\n")
        f.write(f"Total samples: {len(labels)}\n")
        f.write("\n")
        
        # Per-class performance
        f.write("Per-class performance:\n")
        for class_id in sorted(np.unique(labels)):
            mask = labels == class_id
            class_correct = (predictions[mask] == class_id).sum()
            class_total = mask.sum()
            class_acc = class_correct / class_total if class_total > 0 else 0
            f.write(f"  {CLASS_NAMES[class_id]:20s}: {class_correct:3d}/{class_total:3d} correct ({class_acc:.2%})\n")
        
        f.write("\nMost confused pairs:\n")
        confusion_pairs = []
        for true_class in sorted(np.unique(labels)):
            for pred_class in sorted(np.unique(predictions)):
                if true_class != pred_class:
                    mask = (labels == true_class) & (predictions == pred_class)
                    count = mask.sum()
                    if count > 0:
                        confusion_pairs.append((count, true_class, pred_class))
        
        confusion_pairs.sort(reverse=True)
        for count, true_class, pred_class in confusion_pairs[:5]:
            f.write(f"  {CLASS_NAMES[true_class]:15s} → {CLASS_NAMES[pred_class]:15s}: {count:3d} errors\n")
        
        f.write("="*60 + "\n")
    
    print("\n✅ Visualization complete!")
    print(f"\n📂 Saved to: {output_dir}/")
    print(f"   📊 {latent_space_file.name}")
    print(f"   📊 {separability_file.name}")
    print(f"   📄 {summary_file.name}")
    print(f"\nTo view: open {output_dir}/")


if __name__ == "__main__":
    main()
