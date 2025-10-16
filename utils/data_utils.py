
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
    "YSO": 2,
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

def load_combined_data(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load data from Brightpn_id_normspec_counts_label_onlylabelled.txt
    Format: source_id, 380 spectral values, count, label
    
    Returns:
        X: (N, 380) spectral data as float32
        y: (N,) class labels as int64
        src_ids: (N,) source IDs as strings
    """
    # Read the file - it has 383 columns: src_id + 380 spectra + count + label
    df = pd.read_csv(data_path, sep=r"\s+", header=None, dtype=str, engine="python")
    
    if df.shape[1] != 383:
        raise ValueError(f"Expected 383 columns (src_id + 380 spectra + count + label), got {df.shape[1]}")
    
    # Extract components
    src_ids = df.iloc[:, 0].astype(str).values  # First column: source IDs
    X = df.iloc[:, 1:381].astype(np.float32).values  # Columns 1-380: spectral data
    labels = df.iloc[:, 382].astype(str).values  # Last column: labels
    
    # Normalize and map labels
    norm_labels = np.array([_normalize_label(label) for label in labels])
    y = np.array([LABEL_MAP.get(label, -1) for label in norm_labels])
    
    # Check for unknown labels
    unknown_mask = y == -1
    if unknown_mask.any():
        unknown_labels = np.unique(norm_labels[unknown_mask])
        raise ValueError(f"Found unknown labels: {unknown_labels}")
    
    return X, y, src_ids

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
        target_names = ["AGN[1]", "HM/LM[2]"]
        return y, printable_map, target_names
    elif mode == "all4":
        printable_map = PRINTABLE_CLASS_ALL4
        target_names = ["AGN[1]", "HM/LM[2]", "YSO/CV[3]", "NS/HMXB/LMXB/NS_BIN[4]"]
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
