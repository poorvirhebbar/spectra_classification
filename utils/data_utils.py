
import pandas as pd
import numpy as np
import torch
from typing import Tuple, List, Optional

# Printable mapping 0..3 -> 1..4 (for user-facing outputs)
PRINTABLE_CLASS_ALL4 = {0: 1, 1: 2, 2: 3, 3: 4}
PRINTABLE_CLASS_BIN = {0: 1, 1: 2}  # when running with classes 1&2 only

# Label mapping (same as in main.py)
LABEL_MAP = {
    "AGN": 0,
    "HM-STAR": 1,
    "LM-STAR": 1,
    "YSO": 1,      # Grouped with HM/LM
    "CV": 2,
    "NS": 3,
    "HMXB": 3,
    "LMXB": 3,
    "NS_BIN": 3,
}

# Normalization & mapping ------------------------------------------------------
def _normalize_label(s: str) -> str:
    """Uppercase, strip, and keep separators for exact matching with LABEL_MAP keys."""
    s = (s or "").strip().upper()
    # Accept synonyms: HMSTAR -> HM-STAR, LMSTAR -> LM-STAR, NSBIN -> NS_BIN
    if s == "HMSTAR": s = "HM-STAR"
    if s == "LMSTAR": s = "LM-STAR"
    if s == "NSBIN":  s = "NS_BIN"
    return s

def load_combined_data(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load data from Brightpn_id_normspec_counts_label_onlylabelled.txt
    Format: source_id, 380 spectral values, count, label
    
    Returns:
        X: (N, 380) spectral data as float32
        y: (N,) class labels as int64
        src_ids: (N,) source IDs as strings
        counts: (N,) count values as float32
    """
    # Read the file - it has 383 columns: src_id + 380 spectra + count + label
    df = pd.read_csv(data_path, sep=r"\s+", header=None, dtype=str, engine="python")
    
    if df.shape[1] != 383:
        raise ValueError(f"Expected 383 columns (src_id + 380 spectra + count + label), got {df.shape[1]}")
    
    # Extract components
    src_ids = df.iloc[:, 0].astype(str).values  # First column: source IDs
    X = df.iloc[:, 1:381].astype(np.float32).values  # Columns 1-380: spectral data
    counts = df.iloc[:, 381].astype(np.float32).values  # Column 381: counts
    labels = df.iloc[:, 382].astype(str).values  # Last column: labels
    
    # Normalize and map labels
    norm_labels = np.array([_normalize_label(label) for label in labels])
    y = np.array([LABEL_MAP.get(label, -1) for label in norm_labels])
    
    # Check for unknown labels
    unknown_mask = y == -1
    if unknown_mask.any():
        unknown_labels = np.unique(norm_labels[unknown_mask])
        raise ValueError(f"Found unknown labels: {unknown_labels}")
    
    return X, y, src_ids, counts

def filter_and_remap(y: np.ndarray, mode: str = "12") -> Tuple[np.ndarray, dict, list]:
    """
    Filter labels and remap them depending on mode.
    mode="12": keep original classes {0,1} -> {0,1} (drop 2,3)
    mode="all4": keep all {0,1,2,3} -> {0,1,2,3}
    Returns:
        y_new: remapped labels
        printable_map: mapping from internal indices to printable classes
        target_names: list of target names for reports
    """
    if mode == "12":
        mask = np.isin(y, [0, 1])
        y = y[mask]
        # Already in {0,1}; keep as-is
        printable_map = PRINTABLE_CLASS_BIN
        target_names = ["AGN[1]", "HM/LM/YSO[2]"]
        return y, printable_map, target_names
    elif mode == "all4":
        printable_map = PRINTABLE_CLASS_ALL4
        target_names = ["AGN[1]", "HM/LM/YSO[2]", "CV[3]", "NS/HMXB/LMXB/NS_BIN[4]"]
        return y, printable_map, target_names
    else:
        raise ValueError("mode must be '12' or 'all4'")

def make_tensors(X: np.ndarray, y: np.ndarray):
    X_t = torch.from_numpy(X).float().unsqueeze(1)  # (N,1,L) - ensure float32
    y_t = torch.from_numpy(y.astype(np.int64))
    return X_t, y_t


def augment_spectrum(X: np.ndarray, noise_std: float = 0.005, shift_range: int = 2) -> np.ndarray:
    """
    Apply minimal augmentation to spectral data (conservative for sensitive spectra).
    
    Args:
        X: (N, features) array of spectra
        noise_std: Standard deviation of Gaussian noise (default: 0.005 = 0.5% of typical values)
        shift_range: Maximum wavelength shift in bins (default: 2)
    
    Returns:
        Augmented spectrum array
    """
    X_aug = X.copy()
    
    # 1. Add very small Gaussian noise (0.5% level - very conservative)
    noise = np.random.normal(0, noise_std, X.shape)
    X_aug = X_aug + noise
    
    # 2. Random small shift (simulate wavelength calibration variations)
    if shift_range > 0:
        shift = np.random.randint(-shift_range, shift_range + 1)
        if shift != 0:
            X_aug = np.roll(X_aug, shift, axis=1)
            # Zero out wrapped-around edges to avoid artifacts
            if shift > 0:
                X_aug[:, :shift] = X_aug[:, shift:shift+1]
            else:
                X_aug[:, shift:] = X_aug[:, shift-1:shift]
    
    return X_aug


def oversample_minority_classes(X: np.ndarray, y: np.ndarray, 
                                 target_count: int = None,
                                 augment: bool = True,
                                 noise_std: float = 0.005) -> Tuple[np.ndarray, np.ndarray]:
    """
    Oversample minority classes to balance the dataset with class-specific augmentation.
    
    Args:
        X: (N, features) feature array
        y: (N,) label array
        target_count: Target samples per class. If None, use majority class count
        augment: Whether to augment duplicated samples
        noise_std: Noise level for augmentation
    
    Returns:
        X_balanced, y_balanced
    """
    from collections import Counter
    
    class_counts = Counter(y)
    if target_count is None:
        target_count = max(class_counts.values())
    
    X_list, y_list = [], []
    
    for class_id in sorted(np.unique(y)):
        mask = y == class_id
        X_class = X[mask]
        y_class = y[mask]
        
        current_count = len(X_class)
        
        if current_count < target_count:
            # Need to oversample
            # Use stronger augmentation for very minority classes (< 100 samples)
            class_noise_std = noise_std * (2.0 if current_count < 100 else 1.5 if current_count < 300 else 1.0)
            class_shift = 3 if current_count < 100 else 2
            
            repeats_needed = (target_count - current_count + current_count - 1) // current_count
            
            # Original samples
            X_list.append(X_class)
            y_list.append(y_class)
            
            # Augmented duplicates
            for _ in range(repeats_needed):
                n_to_add = min(current_count, target_count - len(X_list[-1]))
                if n_to_add > 0:
                    indices = np.random.choice(current_count, n_to_add, replace=False)
                    X_dup = X_class[indices]
                    
                    if augment:
                        # Apply class-specific stronger augmentation
                        X_dup = augment_spectrum(X_dup, noise_std=class_noise_std, shift_range=class_shift)
                    
                    X_list.append(X_dup)
                    y_list.append(y_class[indices])
                    
                    if len(np.concatenate(y_list)) >= target_count * len(np.unique(y)):
                        break
        else:
            # Majority class - keep as is (or downsample if desired)
            X_list.append(X_class)
            y_list.append(y_class)
    
    X_balanced = np.vstack(X_list)
    y_balanced = np.hstack(y_list)
    
    return X_balanced, y_balanced


# ============================================================================
# Semi-supervised learning utilities: Load data with unlabelled samples
# ============================================================================

def load_data_with_unlabelled(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load data file that contains BOTH labelled and unlabelled samples.
    
    File format: source_id, 380 spectral values, count, label
    - If label column is present and valid -> labelled
    - If label column is empty/NaN/"" -> unlabelled
    
    Args:
        data_path: Path to data file (e.g., "data/Brightpn_id_normspec_counts_label.txt")
    
    Returns:
        X_labelled: (N_lab, 380) labelled spectral data
        y_labelled: (N_lab,) labelled class indices
        src_ids_labelled: (N_lab,) source IDs for labelled data
        counts_labelled: (N_lab,) counts for labelled data
        X_unlabelled: (N_unlab, 380) unlabelled spectral data
        src_ids_unlabelled: (N_unlab,) source IDs for unlabelled data
        counts_unlabelled: (N_unlab,) counts for unlabelled data
    """
    # Read file - handle both 382 (no label) and 383 (with label) columns
    df = pd.read_csv(data_path, sep=r"\s+", header=None, dtype=str, engine="python")
    
    # Extract components
    src_ids = df.iloc[:, 0].astype(str).values
    X = df.iloc[:, 1:-2].astype(np.float32).values  # All spectral columns except last 2
    counts = df.iloc[:, -2].astype(np.float32).values  # Second to last column is count
    
    # Last column is label (may be NaN, empty, or valid)
    if df.shape[1] >= 383:
        labels_raw = df.iloc[:, -1].astype(str).values
    else:
        # No label column at all -> all unlabelled
        labels_raw = np.array([''] * len(df))
    
    # Identify labelled vs unlabelled
    # Unlabelled if: NaN, empty string, "nan", "NONE", etc.
    is_labelled = np.array([
        (label not in ['', 'nan', 'NaN', 'NONE', 'None']) and (label != 'nan')
        for label in labels_raw
    ])
    
    # Split into labelled and unlabelled
    X_labelled = X[is_labelled]
    labels_labelled = labels_raw[is_labelled]
    src_ids_labelled = src_ids[is_labelled]
    counts_labelled = counts[is_labelled]
    
    X_unlabelled = X[~is_labelled]
    src_ids_unlabelled = src_ids[~is_labelled]
    counts_unlabelled = counts[~is_labelled]
    
    # Process labelled data through label mapping
    if len(X_labelled) > 0:
        norm_labels = np.array([_normalize_label(label) for label in labels_labelled])
        y_labelled = np.array([LABEL_MAP.get(label, -1) for label in norm_labels])
        
        # Check for unknown labels
        unknown_mask = y_labelled == -1
        if unknown_mask.any():
            unknown_labels = np.unique(norm_labels[unknown_mask])
            raise ValueError(f"Found unknown labels in labelled data: {unknown_labels}")
    else:
        y_labelled = np.array([], dtype=np.int64)
    
    print(f"Loaded from {data_path}:")
    print(f"  Labelled samples: {len(X_labelled)}")
    print(f"  Unlabelled samples: {len(X_unlabelled)}")
    
    return X_labelled, y_labelled, src_ids_labelled, counts_labelled, X_unlabelled, src_ids_unlabelled, counts_unlabelled


def create_test_set(X, y, src_ids, counts=None, n_samples_total=10, random_state=42):
    """
    Create a held-out test set with stratified sampling.
    
    Samples n_samples_total samples in total (not per class), distributed
    proportionally across classes to maintain class balance.
    
    Args:
        X: Feature array (N, 380)
        y: Label array (N,)
        src_ids: Source IDs (N,)
        counts: Count values (N,) - optional
        n_samples_total: Total number of samples to hold out (default: 10)
        random_state: Random seed
    
    Returns:
        X_remaining, y_remaining, src_ids_remaining, counts_remaining: Data after removing test samples
        X_test, y_test, src_ids_test, counts_test: Held-out test data
    
    Example:
        For 4-class with n_samples_total=10:
        - If classes are balanced: ~2-3 samples per class
        - If imbalanced: proportional to class distribution
    """
    np.random.seed(random_state)
    
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)
    
    # Compute class proportions
    class_counts = np.array([np.sum(y == c) for c in unique_classes])
    class_proportions = class_counts / len(y)
    
    # Allocate samples proportionally (at least 1 per class if possible)
    samples_per_class = np.maximum(1, np.round(class_proportions * n_samples_total).astype(int))
    
    # Adjust to exactly n_samples_total
    while samples_per_class.sum() > n_samples_total:
        # Remove from largest allocation
        max_idx = np.argmax(samples_per_class)
        samples_per_class[max_idx] -= 1
    
    while samples_per_class.sum() < n_samples_total:
        # Add to smallest allocation (but not exceeding class size)
        for idx in np.argsort(samples_per_class):
            if samples_per_class[idx] < class_counts[idx]:
                samples_per_class[idx] += 1
                if samples_per_class.sum() >= n_samples_total:
                    break
    
    # Sample from each class
    test_indices = []
    for class_id, n_samples in zip(unique_classes, samples_per_class):
        class_indices = np.where(y == class_id)[0]
        n_samples = min(n_samples, len(class_indices))
        
        if n_samples > 0:
            selected = np.random.choice(class_indices, n_samples, replace=False)
            test_indices.extend(selected)
    
    # Create masks
    test_mask = np.zeros(len(X), dtype=bool)
    test_mask[test_indices] = True
    
    # Split
    X_test = X[test_mask]
    y_test = y[test_mask]
    src_ids_test = src_ids[test_mask]
    
    X_remaining = X[~test_mask]
    y_remaining = y[~test_mask]
    src_ids_remaining = src_ids[~test_mask]
    
    # Handle counts if provided
    if counts is not None:
        counts_test = counts[test_mask]
        counts_remaining = counts[~test_mask]
    else:
        counts_test = None
        counts_remaining = None
    
    return X_remaining, y_remaining, src_ids_remaining, counts_remaining, X_test, y_test, src_ids_test, counts_test


def create_test_set_unlabelled(X, src_ids, n_samples=10, random_state=42):
    """
    Create a held-out test set from unlabelled data (random sampling).
    
    Args:
        X: Feature array (N, 380)
        src_ids: Source IDs (N,)
        n_samples: Number of samples to hold out
        random_state: Random seed
    
    Returns:
        X_remaining, src_ids_remaining: Data after removing test samples
        X_test, src_ids_test: Held-out test data
    """
    np.random.seed(random_state)
    
    n_samples = min(n_samples, len(X))
    test_indices = np.random.choice(len(X), n_samples, replace=False)
    
    test_mask = np.zeros(len(X), dtype=bool)
    test_mask[test_indices] = True
    
    X_test = X[test_mask]
    src_ids_test = src_ids[test_mask]
    
    X_remaining = X[~test_mask]
    src_ids_remaining = src_ids[~test_mask]
    
    return X_remaining, src_ids_remaining, X_test, src_ids_test


def save_test_set(X_test, y_test, src_ids_test, save_path, class_names=None):
    """
    Save test set to JSON for later evaluation.
    
    Args:
        X_test: Test features (N, 380)
        y_test: Test labels (N,) - can be None for unlabelled
        src_ids_test: Source IDs (N,)
        save_path: Path to save JSON file
        class_names: Optional dict mapping class_id -> name
    """
    import json
    
    data = {
        'n_samples': len(X_test),
        'samples': []
    }
    
    for i in range(len(X_test)):
        sample = {
            'source_id': str(src_ids_test[i]),
            'spectrum': X_test[i].tolist(),
        }
        
        if y_test is not None:
            sample['true_label'] = int(y_test[i])
            if class_names is not None:
                sample['true_label_name'] = class_names.get(int(y_test[i]), 'Unknown')
        
        data['samples'].append(sample)
    
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved test set with {len(X_test)} samples to {save_path}")


def load_test_set(load_path):
    """
    Load test set from JSON.
    
    Args:
        load_path: Path to JSON file
    
    Returns:
        X_test, y_test, src_ids_test
        (y_test is None if test set is unlabelled)
    """
    import json
    
    with open(load_path, 'r') as f:
        data = json.load(f)
    
    n_samples = data['n_samples']
    X_test = np.array([sample['spectrum'] for sample in data['samples']], dtype=np.float32)
    src_ids_test = np.array([sample['source_id'] for sample in data['samples']])
    
    # Check if labels exist
    if 'true_label' in data['samples'][0]:
        y_test = np.array([sample['true_label'] for sample in data['samples']], dtype=np.int64)
    else:
        y_test = None
    
    print(f"Loaded test set with {n_samples} samples from {load_path}")
    
    return X_test, y_test, src_ids_test
