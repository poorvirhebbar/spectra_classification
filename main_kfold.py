import argparse, json
import numpy as np
import os
from collections import Counter
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from utils.model_utils import CNN1D, FocalLoss, train_one_epoch, evaluate
from utils.data_utils import (load_combined_data, filter_and_remap, make_tensors,
                               oversample_minority_classes,
                               load_indices, load_data_with_row_indices)

torch.manual_seed(42); np.random.seed(42)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)


# Human-readable names (match your grouping)
CLASS_NAMES = {0: "AGN", 1: "HM/LM/YSO", 2: "CV", 3: "NS/HMXB/LMXB/NS_BIN"}


def parse_args():
    ap = argparse.ArgumentParser(description="K-fold cross validation variant of main.py. "
                                             "The test set is held out; the train set is split into K folds.")
    ap.add_argument("--classes", choices=["2", "4"], default="2",
                    help="'2' = AGN vs HM/LM/YSO (default), '4' = 4-class: AGN, HM/LM/YSO, CV, NS.")
    ap.add_argument("--n_folds", type=int, default=5,
                    help="Number of cross-validation folds (default: 5).")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--oversample_target", type=float, default=0.7,
                    help="Target ratio for minority class oversampling (0.7 = 70% of majority)")
    ap.add_argument("--focal_gamma", type=float, default=2.0,
                    help="Focal loss gamma parameter (higher = more focus on hard examples)")
    ap.add_argument("--minority_boost", type=float, default=1.0,
                    help="Extra weight multiplier for CV/NS classes in 4-class mode (default: 1.0)")
    ap.add_argument("--val_split", type=float, default=0.2,
                    help="Only used (to carve off a test set) when --use_indices is NOT set.")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", default="checkpoints_kfold",
                    help="Directory to save per-fold best checkpoints and the CV summary JSON.")
    ap.add_argument("--data", choices=["Brightmos", "Brightpn"], default="Brightpn",
                    help="Dataset to use: 'Brightmos' or 'Brightpn' (default: Brightpn)")
    ap.add_argument("--data_dir", default="data",
                    help="Directory containing data files (default: data)")
    ap.add_argument("--use_indices", action="store_true",
                    help="Use predefined train/test indices from --indices_dir. The Test_* file is "
                         "held out; the Train_* file is the pool split into K folds.")
    ap.add_argument("--indices_dir", default="data/data_indices",
                    help="Directory holding Train_*/Test_* index files (default: data/data_indices)")
    return ap.parse_args()


def _index_file_paths(indices_dir: str, data: str, classes: str):
    """Resolve the train/test index file paths for the given dataset and class config."""
    if data == "Brightmos":
        train_name = f"Train_MOS_index_{classes}class.txt"
        test_name = f"Test_MOS_index_{classes}class.txt"
    elif data == "Brightpn":
        train_name = f"Train_PN_indices_{classes}class.txt"
        test_name = f"Test_PN_indices_{classes}class.txt"
    else:
        raise ValueError(f"Unknown data choice: {data}")
    return f"{indices_dir}/{train_name}", f"{indices_dir}/{test_name}"


def norm_per_sample(A):
    """Per-spectrum normalization (zero-mean, unit-std per row). Depends only on the row itself."""
    mu = A.mean(axis=1, keepdims=True)
    sigma = A.std(axis=1, keepdims=True) + 1e-8
    return (A - mu) / sigma


def load_pool_and_test(args, mode, num_classes):
    """Return (X_pool, y_pool, X_test, y_test, target_names).

    X_pool/y_pool is the cross-validation pool (split into K folds downstream);
    X_test/y_test is the held-out test set, never touched during fold training.
    """
    if args.use_indices:
        data_file = f"{args.data_dir}/{args.data}_id_normspec_counts_label.txt"
        print(f"Loading data from {data_file} (use_indices=True)...")
        (X_lab, y_lab, src_ids_lab, counts_lab, row_idx_lab,
         _Xu, _su, _cu, _ru) = load_data_with_row_indices(data_file)
        print(f"Loaded {len(y_lab)} labelled spectra, each with {X_lab.shape[1]} bins")

        vals, cnts = np.unique(y_lab, return_counts=True)
        print(f"Label distribution (all labelled): "
              f"{ {CLASS_NAMES[int(v)]: int(c) for v, c in zip(vals, cnts)} }")

        _, _printable_map, target_names = filter_and_remap(y_lab, mode)

        train_idx_path, test_idx_path = _index_file_paths(args.indices_dir, args.data, args.classes)
        print(f"Loading indices: {train_idx_path}, {test_idx_path}")
        train_indices = load_indices(train_idx_path)
        test_indices = load_indices(test_idx_path)
        print(f"  train(pool) indices: {len(train_indices)}; test indices: {len(test_indices)}")

        train_mask = np.isin(row_idx_lab, train_indices)
        test_mask = np.isin(row_idx_lab, test_indices)

        X_pool, y_pool = X_lab[train_mask], y_lab[train_mask]
        X_test, y_test = X_lab[test_mask], y_lab[test_mask]

        if args.classes == "2":
            m_pool = np.isin(y_pool, [0, 1])
            X_pool, y_pool = X_pool[m_pool], y_pool[m_pool]
            m_test = np.isin(y_test, [0, 1])
            X_test, y_test = X_test[m_test], y_test[m_test]
    else:
        data_file = f"{args.data_dir}/{args.data}_id_normspec_counts_label_onlylabelled.txt"
        print(f"Loading data from {data_file}...")
        X_all, y_all, _src_ids, _counts = load_combined_data(data_file)
        print(f"Loaded {len(y_all)} labeled spectra, each with {X_all.shape[1]} bins")

        y_remapped, _printable_map, target_names = filter_and_remap(y_all, mode)
        if args.classes == "2":
            mask = np.isin(y_all, [0, 1])
            X_filtered = X_all[mask]
            y_filtered = y_remapped
        else:
            X_filtered = X_all
            y_filtered = y_all

        # Carve off a held-out test set (same stratified split convention as main.py)
        X_pool, X_test, y_pool, y_test = train_test_split(
            X_filtered, y_filtered, test_size=args.val_split,
            random_state=42, stratify=y_filtered
        )

    return X_pool, y_pool, X_test, y_test, target_names


def train_fold(args, fold, num_classes, target_names,
               Xtr, ytr, Xva, yva, test_loader, y_test, device):
    """Train one fold from scratch.

    Returns (train_acc, val_acc, test_acc, best_epoch, ckpt_path) where ckpt_path
    is the saved best-by-val-accuracy checkpoint for this fold.
    """
    # Reproducible-but-distinct init per fold
    torch.manual_seed(42 + fold)
    np.random.seed(42 + fold)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42 + fold)

    # Oversample minority classes (4-class only) on the TRAIN fold only
    if args.classes == "4":
        class_counts = Counter(ytr)
        target_count = int(max(class_counts.values()) * args.oversample_target)
        Xtr, ytr = oversample_minority_classes(Xtr, ytr, target_count=target_count,
                                               augment=True, noise_std=0.005)
        print(f"    [fold {fold}] after oversampling: {dict(zip(*np.unique(ytr, return_counts=True)))}")

    # Per-sample normalization (train + val of this fold)
    Xtr = norm_per_sample(Xtr)
    Xva = norm_per_sample(Xva)

    Xtr_t, ytr_t = make_tensors(Xtr, ytr)
    Xva_t, yva_t = make_tensors(Xva, yva)

    # Class weights from THIS fold's training labels
    class_counts = np.bincount(ytr, minlength=num_classes)
    class_weights = (class_counts.sum() / (class_counts + 1e-8)).astype(np.float32)
    if num_classes == 4 and args.minority_boost > 1.0:
        class_weights[2] *= args.minority_boost  # CV
        class_weights[3] *= args.minority_boost  # NS/HMXB/etc
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32)

    # Balanced sampler for training
    ex_weights = class_weights_t[ytr_t].cpu().double()
    sampler = WeightedRandomSampler(weights=ex_weights,
                                    num_samples=len(ex_weights),
                                    replacement=True)
    train_loader = DataLoader(TensorDataset(Xtr_t, ytr_t),
                              batch_size=args.batch_size, sampler=sampler,
                              shuffle=False, num_workers=args.num_workers)
    train_eval_loader = DataLoader(TensorDataset(Xtr_t, ytr_t),
                                   batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.num_workers)
    val_loader = DataLoader(TensorDataset(Xva_t, yva_t),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    # Model / optim / loss
    model = CNN1D(num_classes=num_classes).to(device)
    criterion = FocalLoss(alpha=class_weights_t.to(device), gamma=args.focal_gamma)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )

    # Track best-by-val-accuracy within this fold
    best = {"acc": -1.0, "epoch": 0, "state": None}
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, criterion, optim, device)
        tr_loss, tr_acc = evaluate(model, train_eval_loader, criterion, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(va_loss)

        if va_acc > best["acc"]:
            best = {"acc": va_acc, "epoch": epoch,
                    "train_acc": tr_acc,
                    "state": {k: v.cpu().clone() for k, v in model.state_dict().items()}}

        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"    [fold {fold}] epoch {epoch:03d} | "
                  f"train_acc={tr_acc:.4f} | val_acc={va_acc:.4f} "
                  f"(best val={best['acc']:.4f}@{best['epoch']})")

    # Restore best-val model, then measure the held-out TEST accuracy
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    _test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    # Always save the fold's best checkpoint so it can be visualized / used for inference
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(
        args.out_dir,
        f"fold{fold}_{args.classes}cls_{args.data}_epoch{best['epoch']:03d}_val{best['acc']:.4f}.pt")
    torch.save({
        "state_dict": best["state"],
        "num_classes": num_classes,
        "class_names": CLASS_NAMES,
        "args": vars(args),
        "fold": fold,
        "best_epoch": best["epoch"],
    }, ckpt_path)
    print(f"    [fold {fold}] saved best checkpoint: {ckpt_path}")

    return best["train_acc"], best["acc"], test_acc, best["epoch"], ckpt_path


def main():
    args = parse_args()
    device = torch.device(args.device)

    if args.classes == "2":
        mode, num_classes = "12", 2
    else:
        mode, num_classes = "all4", 4

    # ---- Load the CV pool and the held-out test set --------------------
    X_pool, y_pool, X_test, y_test, target_names = load_pool_and_test(args, mode, num_classes)

    if len(np.unique(y_pool)) < num_classes:
        print(f"Pool only has classes {sorted(np.unique(y_pool).tolist())} "
              f"but num_classes={num_classes}. Exiting.")
        return

    print(f"\nCV pool: {len(y_pool)} samples | Held-out test: {len(y_test)} samples | "
          f"{num_classes} classes")
    print(f"Pool distribution: {dict(zip(*[a.tolist() for a in np.unique(y_pool, return_counts=True)]))}")
    print(f"Test distribution: {dict(zip(*[a.tolist() for a in np.unique(y_test, return_counts=True)]))}")

    # Test set: normalize once (per-sample, so it's split-independent) and reuse for every fold
    X_test_norm = norm_per_sample(X_test)
    Xte_t, yte_t = make_tensors(X_test_norm, y_test)
    test_loader = DataLoader(TensorDataset(Xte_t, yte_t),
                             batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    # ---- K-fold over the pool -----------------------------------------
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)

    # Length-K accuracy ranges, one entry per fold
    train_accuracies = []
    val_accuracies = []
    test_accuracies = []
    best_epochs = []
    ckpt_paths = []

    print(f"\n{'='*70}\nRunning {args.n_folds}-fold cross validation\n{'='*70}")
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_pool, y_pool)):
        print(f"\n----- Fold {fold + 1}/{args.n_folds} "
              f"(train={len(tr_idx)}, val={len(val_idx)}) -----")
        tr_acc, va_acc, te_acc, best_epoch, ckpt_path = train_fold(
            args, fold, num_classes, target_names,
            X_pool[tr_idx], y_pool[tr_idx],
            X_pool[val_idx], y_pool[val_idx],
            test_loader, y_test, device,
        )
        train_accuracies.append(tr_acc)
        val_accuracies.append(va_acc)
        test_accuracies.append(te_acc)
        best_epochs.append(best_epoch)
        ckpt_paths.append(ckpt_path)
        print(f"  >> Fold {fold + 1} done | train_acc={tr_acc:.4f} | "
              f"val_acc={va_acc:.4f} | test_acc={te_acc:.4f} (best epoch {best_epoch})")

    # ---- Summary -------------------------------------------------------
    train_accuracies = np.array(train_accuracies)
    val_accuracies = np.array(val_accuracies)
    test_accuracies = np.array(test_accuracies)

    print(f"\n{'='*70}\n{args.n_folds}-Fold CV summary "
          f"({args.data}, {args.classes}-class)\n{'='*70}")
    print(f"{'fold':>5} | {'train_acc':>10} | {'val_acc':>8} | {'test_acc':>8} | {'best_ep':>7}")
    print("-" * 70)
    for i in range(args.n_folds):
        print(f"{i + 1:>5} | {train_accuracies[i]:>10.4f} | {val_accuracies[i]:>8.4f} | "
              f"{test_accuracies[i]:>8.4f} | {best_epochs[i]:>7d}")
    print("-" * 70)
    print(f"{'mean':>5} | {train_accuracies.mean():>10.4f} | {val_accuracies.mean():>8.4f} | "
          f"{test_accuracies.mean():>8.4f} |")
    print(f"{'std':>5} | {train_accuracies.std():>10.4f} | {val_accuracies.std():>8.4f} | "
          f"{test_accuracies.std():>8.4f} |")

    # The "best fold" = highest validation accuracy; its checkpoint is used for visualization/inference
    best_fold = int(np.argmax(val_accuracies))
    best_checkpoint = os.path.abspath(ckpt_paths[best_fold])
    print(f"\nBest fold: {best_fold + 1} (val_acc={val_accuracies[best_fold]:.4f}, "
          f"test_acc={test_accuracies[best_fold]:.4f})")
    print(f"Best checkpoint: {best_checkpoint}")

    # Persist the ranges so they can be loaded later
    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(
        args.out_dir, f"cv_summary_{args.data}_{args.classes}cls_{args.n_folds}fold.json")
    with open(summary_path, "w") as f:
        json.dump({
            "data": args.data,
            "classes": args.classes,
            "mode": "supervised",
            "n_folds": args.n_folds,
            "train_accuracies": train_accuracies.tolist(),
            "val_accuracies": val_accuracies.tolist(),
            "test_accuracies": test_accuracies.tolist(),
            "best_epochs": best_epochs,
            "fold_checkpoints": [os.path.abspath(p) for p in ckpt_paths],
            "best_fold": best_fold,
            "best_checkpoint": best_checkpoint,
            "train_acc_mean": float(train_accuracies.mean()),
            "train_acc_std": float(train_accuracies.std()),
            "val_acc_mean": float(val_accuracies.mean()),
            "val_acc_std": float(val_accuracies.std()),
            "test_acc_mean": float(test_accuracies.mean()),
            "test_acc_std": float(test_accuracies.std()),
            "args": vars(args),
        }, f, indent=2)
    print(f"\nSaved CV summary to {summary_path}")


if __name__ == "__main__":
    main()
