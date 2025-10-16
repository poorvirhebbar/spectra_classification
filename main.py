import argparse, json
import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from utils.model_utils import CNN1D, FocalLoss, train_one_epoch, evaluate
from utils.data_utils import (load_combined_data, filter_and_remap, make_tensors, 
                               oversample_minority_classes)

torch.manual_seed(42); np.random.seed(42)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)


# Human-readable names (match your grouping)
CLASS_NAMES = {0: "AGN", 1: "HM/LM", 2: "YSO/CV", 3: "NS/HMXB/LMXB/NS_BIN"}



def parse_args():
    ap = argparse.ArgumentParser(description="Train a spectra classifier from combined data file.")
    ap.add_argument("--classes", choices=["2", "4"], default="2",
                    help="'2' = AGN vs HM/LM (default), '4' = 4-class.")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--oversample_target", type=float, default=0.7,
                    help="Target ratio for minority class oversampling (0.7 = 70% of majority)")
    ap.add_argument("--focal_gamma", type=float, default=2.0,
                    help="Focal loss gamma parameter (higher = more focus on hard examples)")
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save_preds", default=None,
                    help="Optional JSON path to save validation predictions.")
    ap.add_argument("--out_dir", default="checkpoints",
                    help="Directory to save model checkpoints (best_*.pt, last_*.pt).")
    return ap.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)

    # Use onlylabelled file for both 2 and 4 classes (the full file contains mostly unlabeled data)
    data_file = "data/Brightpn_id_normspec_counts_label_onlylabelled.txt"

    # Load data from single file
    print(f"Loading data from {data_file}...")
    X_all, y_all, src_ids = load_combined_data(data_file)
    
    print(f"Loaded {len(y_all)} labeled spectra, each with {X_all.shape[1]} bins")
    
    # Debug: distribution
    vals, cnts = np.unique(y_all, return_counts=True)
    dist = {CLASS_NAMES[int(v)]: int(c) for v, c in zip(vals, cnts)}
    print(f"Label distribution: {dist}")

    # Filter and remap based on class mode
    if args.classes == "2":
        mode = "12"
        num_classes = 2
    else:
        mode = "all4"
        num_classes = 4
    
    y_filtered, printable_map, target_names = filter_and_remap(y_all, mode)
    
    # Apply the same filtering to X
    if args.classes == "2":
        mask = np.isin(y_all, [0, 1])
        X_filtered = X_all[mask]
    else:
        X_filtered = X_all
        y_filtered = y_all

    if len(y_filtered) == 0:
        print("No data to train on after class filtering. Exiting.")
        return
    if len(np.unique(y_filtered)) < num_classes:
        print(f"Only found classes {sorted(np.unique(y_filtered).tolist())} after filtering, "
              f"but num_classes={num_classes}. Exiting.")
        return

    print(f"Training on {len(y_filtered)} labeled spectra across {num_classes} classes.")

    # Split data
    Xtr, Xva, ytr, yva = train_test_split(X_filtered, y_filtered, test_size=args.val_split,
                                          random_state=42, stratify=y_filtered)
    
    # Apply data augmentation and oversampling for minority classes (4-class mode only)
    if args.classes == "4":
        print(f"Original training distribution: {dict(zip(*np.unique(ytr, return_counts=True)))}")
        # Oversample minority classes with augmentation
        from collections import Counter
        class_counts = Counter(ytr)
        target_count = int(max(class_counts.values()) * args.oversample_target)
        Xtr, ytr = oversample_minority_classes(Xtr, ytr, target_count=target_count, 
                                                augment=True, noise_std=0.005)
        print(f"After oversampling (target={args.oversample_target:.0%} of majority): {dict(zip(*np.unique(ytr, return_counts=True)))}")
        print(f"Training set size: {len(ytr)} (increased from {len(X_filtered) * (1 - args.val_split):.0f})")
                        
    # Per-spectrum normalization (zero-mean, unit-std per row)
    def norm_per_sample(A):
        mu = A.mean(axis=1, keepdims=True)
        sigma = A.std(axis=1, keepdims=True) + 1e-8
        return (A - mu) / sigma

    Xtr = norm_per_sample(Xtr)
    Xva = norm_per_sample(Xva)

    # 6) Tensors + loaders  ---------------------------------------------
    # Convert to tensors (adds channel dim)
    Xtr_t, ytr_t = make_tensors(Xtr, ytr)
    Xva_t, yva_t = make_tensors(Xva, yva)

    # Compute class weights from **training** labels (helps minority recall)
    class_counts = np.bincount(ytr, minlength=num_classes)
    class_weights = (class_counts.sum() / (class_counts + 1e-8)).astype(np.float32)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32)  # CPU tensor for sampler; moved to device for loss

    # Balanced sampling: weight per-example = weight of its class
    train_ds = TensorDataset(Xtr_t, ytr_t)
    ex_weights = class_weights_t[ytr_t].cpu().double()   # shape: (N_train,)
    sampler = WeightedRandomSampler(weights=ex_weights,
                                    num_samples=len(ex_weights),
                                    replacement=True)                                                 

    train_loader = DataLoader(train_ds,
                              batch_size=args.batch_size,
                              sampler=sampler,   # no shuffle when using sampler
                              shuffle=False,
                              num_workers=args.num_workers)

    # Separate, deterministic loader for measuring TRAIN accuracy (no sampler)
    train_eval_loader = DataLoader(TensorDataset(Xtr_t, ytr_t),
                                   batch_size=args.batch_size,
                                   shuffle=False,
                                   num_workers=args.num_workers)

    val_loader = DataLoader(TensorDataset(Xva_t, yva_t),
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers)

    # 7) Model/optim  ----------------------------------------------------
    model = CNN1D(num_classes=num_classes).to(device)
    
    # Use Focal Loss for better handling of class imbalance (especially for 4-class)
    criterion = FocalLoss(alpha=class_weights_t.to(device), gamma=args.focal_gamma)
    
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler: reduces LR when validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode='min', factor=0.5, patience=10, verbose=True, min_lr=1e-6
    )

    # Train + checkpoints
    os.makedirs(args.out_dir, exist_ok=True)
    best = {"acc": -1.0, "state": None}
    tag = f"{args.classes}cls_{len(y_filtered)}ex_{X_filtered.shape[1]}bins"

    for epoch in range(1, args.epochs + 1):
        tr_loss_epoch = train_one_epoch(model, train_loader, criterion, optim, device)

        # Measure TRAIN & VAL on fixed loaders (no sampler)
        tr_loss_eval, tr_acc_eval = evaluate(model, train_eval_loader, criterion, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch:03d} | "
              f"train_loss={tr_loss_epoch:.4f} | train_acc={tr_acc_eval:.4f} | "
              f"val_loss={va_loss:.4f} | val_acc={va_acc:.4f}")
        
        # Step learning rate scheduler based on validation loss
        scheduler.step(va_loss)

        # Save BEST checkpoint by val_acc
        if va_acc > best["acc"]:
            best = {"acc": va_acc, "state": {k: v.cpu() for k, v in model.state_dict().items()}}
            best_epoch = epoch
            best_ckpt_path = os.path.join(args.out_dir, f"best_{tag}_val{va_acc:.4f}.pt")
            torch.save({
                "state_dict": best["state"],
                "num_classes": num_classes,
                "class_names": CLASS_NAMES,
                "args": vars(args),
                "best_epoch": best_epoch,
            }, best_ckpt_path)
            print(f"  ↳ Saved BEST checkpoint: {best_ckpt_path} (val_acc={va_acc:.4f})")

    # Load best back (optional but recommended before final eval)
    if best["state"] is not None:
        model.load_state_dict(best["state"])
        model.to(device)

    # Also save LAST checkpoint for reproducibility
    last_ckpt_path = os.path.join(args.out_dir, f"last_{tag}_val{va_acc:.4f}.pt")
    torch.save({
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "num_classes": num_classes,
        "class_names": CLASS_NAMES,
        "args": vars(args),
    }, last_ckpt_path)
    print(f"Saved LAST checkpoint: {last_ckpt_path}")


    # 9) Final validation metrics ----------------------------------------------------
    model.eval()
    with torch.no_grad():
        y_pred = []
        for xb, _ in val_loader:
            xb = xb.to(device)
            logits = model(xb)
            y_pred.append(torch.argmax(logits, dim=1).cpu().numpy())
        y_pred = np.concatenate(y_pred) if y_pred else np.array([], dtype=int)

    acc = accuracy_score(yva, y_pred)
    print("\n=== Validation Metrics ===")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(yva, y_pred, target_names=target_names, digits=4))
    print("Confusion matrix:\n", confusion_matrix(yva, y_pred))

    if args.save_preds:
        with open(args.save_preds, "w") as f:
            json.dump({"preds": y_pred.tolist()}, f, indent=2)
        print(f"Saved validation predictions to {args.save_preds}")

if __name__ == "__main__":
    main()
