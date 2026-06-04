"""
Inference for k-fold checkpoints (supervised: main_kfold.py, semi-supervised: main_unlabelled_kfold.py).

Loads a checkpoint (directly, or the best fold recorded in a CV summary JSON), applies the
SAME per-sample normalization used at training time, and runs inference. By default it
evaluates on the held-out TEST indices for the checkpoint's dataset/class config and prints a
classification report + confusion matrix. With --data_file it instead predicts on an arbitrary
spectra file (labels optional).

Examples
--------
# Evaluate the best fold recorded in a CV summary on its held-out test set:
python infer_kfold.py --summary checkpoints_kfold/Brightpn_2cls_supervised/cv_summary_Brightpn_2cls_5fold.json

# Evaluate a specific checkpoint on the test indices:
python infer_kfold.py --checkpoint checkpoints_kfold/.../fold2_2cls_Brightpn_epoch042_val0.93.pt

# Predict on an arbitrary spectra file and save predictions:
python infer_kfold.py --checkpoint <ckpt> --data_file data/some_spectra.txt --save_preds preds.json
"""

import argparse, json
import numpy as np
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from utils.model_utils import CNN1D
from utils.data_utils import (filter_and_remap, make_tensors, load_indices,
                              load_data_with_row_indices, load_combined_data)

CLASS_NAMES = {0: "AGN", 1: "HM/LM/YSO", 2: "CV", 3: "NS/HMXB/LMXB/NS_BIN"}


def norm_per_sample(A):
    """Per-spectrum z-score (zero-mean, unit-std per row) — identical to training preprocessing."""
    mu = A.mean(axis=1, keepdims=True)
    sigma = A.std(axis=1, keepdims=True) + 1e-8
    return (A - mu) / sigma


def _index_file_paths(indices_dir, data, classes):
    if data == "Brightmos":
        return (f"{indices_dir}/Train_MOS_index_{classes}class.txt",
                f"{indices_dir}/Test_MOS_index_{classes}class.txt")
    elif data == "Brightpn":
        return (f"{indices_dir}/Train_PN_indices_{classes}class.txt",
                f"{indices_dir}/Test_PN_indices_{classes}class.txt")
    raise ValueError(f"Unknown data choice: {data}")


def parse_args():
    ap = argparse.ArgumentParser(description="Run inference with a k-fold checkpoint.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="Path to a checkpoint .pt file.")
    src.add_argument("--summary", help="Path to a cv_summary_*.json; uses its 'best_checkpoint'.")

    ap.add_argument("--classes", choices=["2", "4"], default=None,
                    help="Override; otherwise auto-detected from the checkpoint.")
    ap.add_argument("--data_choice", choices=["Brightmos", "Brightpn"], default=None,
                    help="Override; otherwise auto-detected from the checkpoint.")
    ap.add_argument("--data_file", default=None,
                    help="Run inference on this spectra file instead of the held-out test indices. "
                         "Uses load_combined_data format (id, spectrum, counts, label).")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--indices_dir", default="data/data_indices")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save_preds", default=None, help="Optional JSON path to write predictions.")
    return ap.parse_args()


def resolve_checkpoint(args):
    if args.summary:
        with open(args.summary) as f:
            summary = json.load(f)
        ckpt = summary.get("best_checkpoint")
        if not ckpt:
            raise SystemExit(f"No 'best_checkpoint' field in {args.summary}")
        print(f"Using best checkpoint from summary (fold {summary.get('best_fold', '?')}): {ckpt}")
        return ckpt
    return args.checkpoint


def main():
    args = parse_args()
    device = torch.device(args.device)

    ckpt_path = resolve_checkpoint(args)
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    print(f"\nLoading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    num_classes = checkpoint["num_classes"]
    ckpt_args = checkpoint.get("args", {}) or {}

    # Auto-detect classes / dataset from the checkpoint, allowing CLI overrides
    if args.classes is None:
        args.classes = ckpt_args.get("classes") or ("4" if num_classes == 4 else "2")
    if args.data_choice is None:
        args.data_choice = ckpt_args.get("data")
        if args.data_choice not in ("Brightmos", "Brightpn"):
            name = Path(ckpt_path).name
            args.data_choice = "Brightmos" if "Brightmos" in name else "Brightpn"
    mode = "12" if args.classes == "2" else "all4"
    print(f"  classes={args.classes} | data={args.data_choice} | num_classes={num_classes}")

    # Build model
    model = CNN1D(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # ---- Build the evaluation set -------------------------------------
    if args.data_file:
        print(f"Loading inference data from {args.data_file}...")
        X, y, _src, _counts = load_combined_data(args.data_file)
        if args.classes == "2":
            m = np.isin(y, [0, 1])
            X, y = X[m], y[m]
        _, _printable, target_names = filter_and_remap(y, mode)
        has_labels = True
    else:
        data_file = f"{args.data_dir}/{args.data_choice}_id_normspec_counts_label.txt"
        print(f"Loading held-out test indices for {args.data_choice} ({args.classes}-class)...")
        (X_lab, y_lab, _s, _c, row_idx_lab, *_rest) = load_data_with_row_indices(data_file)
        _, _printable, target_names = filter_and_remap(y_lab, mode)
        _train_idx_path, test_idx_path = _index_file_paths(args.indices_dir, args.data_choice, args.classes)
        test_indices = load_indices(test_idx_path)
        test_mask = np.isin(row_idx_lab, test_indices)
        X, y = X_lab[test_mask], y_lab[test_mask]
        if args.classes == "2":
            m = np.isin(y, [0, 1])
            X, y = X[m], y[m]
        has_labels = True
        print(f"  {len(y)} held-out test samples")

    # CRITICAL: same per-sample normalization as training
    X = norm_per_sample(X)
    X_t, y_t = make_tensors(X, y)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=args.batch_size, shuffle=False)

    # ---- Inference ----------------------------------------------------
    all_preds, all_probs = [], []
    with torch.no_grad():
        for xb, _ in loader:
            logits = model(xb.to(device))
            probs = torch.softmax(logits, dim=1)
            all_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    preds = np.concatenate(all_preds)
    probs = np.vstack(all_probs)

    # ---- Report -------------------------------------------------------
    print("\n" + "=" * 70)
    if has_labels:
        acc = accuracy_score(y, preds)
        print(f"Accuracy: {acc:.4f}  ({(preds == y).sum()}/{len(y)})")
        print("\nClassification report:")
        print(classification_report(y, preds, target_names=target_names, digits=4))
        print("Confusion matrix:\n", confusion_matrix(y, preds))
    else:
        print(f"Predicted {len(preds)} samples (no labels available).")
    print("=" * 70)

    if args.save_preds:
        out = {
            "checkpoint": os.path.abspath(ckpt_path),
            "classes": args.classes,
            "data": args.data_choice,
            "predictions": [
                {
                    "index": int(i),
                    "predicted": int(preds[i]),
                    "predicted_name": CLASS_NAMES[int(preds[i])],
                    "confidence": float(probs[i, preds[i]]),
                    **({"true_label": int(y[i]),
                        "true_label_name": CLASS_NAMES[int(y[i])],
                        "correct": bool(y[i] == preds[i])} if has_labels else {}),
                }
                for i in range(len(preds))
            ],
        }
        if has_labels:
            out["accuracy"] = float(accuracy_score(y, preds))
        with open(args.save_preds, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved predictions to {args.save_preds}")


if __name__ == "__main__":
    main()
