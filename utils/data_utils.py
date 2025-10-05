
import pandas as pd
import numpy as np
import torch
from typing import Tuple, List, Optional

# Printable mapping 0..3 -> 1..4 (for user-facing outputs)
PRINTABLE_CLASS_ALL4 = {0: 1, 1: 2, 2: 3, 3: 4}
PRINTABLE_CLASS_BIN = {0: 1, 1: 2}  # when running with classes 1&2 only

# Normalization & mapping ------------------------------------------------------
def _normalize_label(s: str) -> str:
    """Uppercase, strip, and remove separators so 'LM-STAR' => 'LMSTAR', 'NS_bin' => 'NSBIN'."""
    s = (s or "").strip().upper()
    # collapse spaces and separators
    s = s.replace("-", "").replace("_", "").replace(" ", "")
    return s

def subset_by_classes(X_labeled, y_all_labeled, allowed=(0, 1)):
    """
    Keep only rows whose label is in `allowed`, drop all others.
    """
    mask = np.isin(y_all_labeled, list(allowed))
    return X_labeled[mask], y_all_labeled[mask]

# Robust readers ---------------------------------------------------------------
def _read_whitespace_table(path: str, n_cols: Optional[int] = None) -> pd.DataFrame:
    """
    Read a whitespace-delimited text file into a DataFrame.
    If n_cols is provided, enforce that many columns (pad/truncate as necessary).
    """
    df = pd.read_csv(path, sep=r"\s+", header=None, dtype=str, engine="python")
    if n_cols is not None:
        # Ensure exactly n_cols columns by adding empty columns if needed
        while df.shape[1] < n_cols:
            df[df.shape[1]] = ""
        if df.shape[1] > n_cols:
            df = df.iloc[:, :n_cols]
    return df

def load_labels_with_ids(labels_txt_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads AllSrc_classes.txt with format:
        <SRC_ID> <LABEL>
    where <LABEL> may be missing (blank). We keep only rows with labels.
    Returns:
        ids_all: (N_all,) ids for every row in the file (including unlabeled)
        ids_labeled: (M,) array of SRC_ID strings for labeled rows
        y: (M,) int array in {0,1,2,3} matching LABEL_MAP
    """
    df = _read_whitespace_table(labels_txt_path, n_cols=2)
    df.columns = ["src_id", "label"]
    ids_all = df["src_id"].astype(str).values
    mask_labeled = df["label"].astype(str).str.strip() != ""
    df_lab = df.loc[mask_labeled].copy()
    if df_lab.empty:
        raise ValueError("No labeled rows found in labels file.")
    norm = df_lab["label"].astype(str).map(_normalize_label)
    mapped = norm.map(LABEL_MAP.get)
    if mapped.isnull().any():
        bad = df_lab.loc[mapped.isnull(), ["src_id", "label"]].head(10)
        raise ValueError(f"Found unknown label values. Example rows:\n{bad.to_string(index=False)}")
    ids = df_lab["src_id"].astype(str).values
    y = mapped.astype(int).values
    return ids_all, ids, y

def load_spectra_matrix(spectra_txt_path: str) -> np.ndarray:
    """
    Reads PN_spectra_rebinned.txt as (N, L) float32 matrix with no header.
    """
    df = pd.read_csv(spectra_txt_path, sep=r"\s+", header=None)
    X = df.values.astype(np.float32)
    if X.ndim != 2:
        raise ValueError("Spectra file must be 2D (N, L).")
    return X

def load_pn_ids(pn_ids_path: str) -> np.ndarray:
    """Reads PN_srcids.txt: one SRC_ID per row, aligned with PN_spectra_rebinned.txt rows."""
    df = pd.read_csv(pn_ids_path, sep=r"\s+", header=None, dtype=str, engine="python")
    return df.iloc[:, 0].astype(str).values

def align_by_labeled_rows(X: np.ndarray, ids_all: np.ndarray, ids_labeled: np.ndarray) -> np.ndarray:
    """
    Assumes rows of X are aligned with the rows of AllSrc_classes.txt (including unlabeled).
    Given the full list of IDs in the labels file (including unlabeled), we select only the rows
    whose IDs are labeled, preserving order.
    """
    id_to_row = {sid: i for i, sid in enumerate(ids_all)}
    rows = [id_to_row[sid] for sid in ids_labeled if sid in id_to_row]
    return X[rows]

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
        target_names = ["AGN[1]", "HM/LM/YSO[2]", "CV[3]", "NS/HMXB/LMXB/NS_bin[4]"]
        return y, printable_map, target_names
    else:
        raise ValueError("mode must be '12' or 'all4'")

def make_tensors(X: np.ndarray, y: np.ndarray):
    X_t = torch.from_numpy(X).unsqueeze(1)  # (N,1,L)
    y_t = torch.from_numpy(y.astype(np.int64))
    return X_t, y_t
