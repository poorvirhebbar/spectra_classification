
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
    X_t = torch.from_numpy(X).unsqueeze(1)  # (N,1,L)
    y_t = torch.from_numpy(y.astype(np.int64))
    return X_t, y_t
