import argparse, json
import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from utils.model_utils import CNN1D, train_one_epoch, evaluate

torch.manual_seed(42); np.random.seed(42)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)


# Human-readable names (match your grouping)
CLASS_NAMES = {0: "AGN", 1: "HM/LM", 2: "YSO/CV", 3: "NS/HMXB/LMXB/NS_BIN"}

# For binary runs, keep ONLY these labels (after normalizing by removing '-' and '_')
BINARY_KEEP = {"AGN", "HMSTAR", "LMSTAR"}  # do NOT include YSO for binary; YSO is class 2

# ----- label normalization & mapping (your spec) -----
def normalize_label_keep_seps(s: str) -> str:
    """Uppercase and keep '-'/'_' (match your exact keys), also accept no-sep variants."""
    s = (s or "").strip().upper()
    # Accept synonyms: HMSTAR -> HM-STAR, LMSTAR -> LM-STAR, NSBIN -> NS_BIN
    if s == "HMSTAR": s = "HM-STAR"
    if s == "LMSTAR": s = "LM-STAR"
    if s == "NSBIN":  s = "NS_BIN"
    return s

LABEL_MAP = {
    "AGN": 0,
    "HM-STAR": 1,
    "LM-STAR": 1,
    "YSO": 2,        # YSO belongs to class 2
    "CV": 2,         # CV also mapped to class 2 per your spec
    "NS": 3,
    "HMXB": 3,
    "LMXB": 3,
    "NS_BIN": 3,
}

def make_tensors(X: np.ndarray, y: np.ndarray):
    X_t = torch.from_numpy(X.astype(np.float32)).unsqueeze(1)  # (N,1,L)
    y_t = torch.from_numpy(y.astype(np.int64))
    return X_t, y_t

def parse_args():
    ap = argparse.ArgumentParser(description="Train a spectra classifier from two row-aligned .txt files.")
    ap.add_argument("--labels",  default="data/Allsrc_classes.txt",
                    help="Two columns: SRC_ID LABEL (LABEL may be blank/None).")
    ap.add_argument("--spectra", default="data/PN_spectra_rebinned.txt",
                    help="(N, L=380) floats; either row-aligned with the full labels file, or already the labeled subset.")
    ap.add_argument("--classes", choices=["2", "4"], default="2",
                    help="'2' = AGN vs HM/LM (default), '4' = 4-class.")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
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

    # 1) Read labels; treat 'None' or empty as unlabeled ----------------------------------------------------
    labels_df_full = pd.read_csv(
        args.labels, sep=r"\s+", header=None, names=["src_id", "label"], dtype=str, engine="python"
    )
    labels_df_full["label"] = labels_df_full["label"].fillna("")
    is_none = labels_df_full["label"].str.strip().str.lower().eq("none")
    labels_df_full.loc[is_none, "label"] = ""

    # 2) Spectra matrix ----------------------------------------------------
    X_full = pd.read_csv(args.spectra, sep=r"\s+", header=None, engine="python").values.astype(np.float32)

    n_labels_total = len(labels_df_full)
    n_spectra = X_full.shape[0]

    # Prepare helper columns used for filtering
    labels_df_full["norm_nosep"] = (
        labels_df_full["label"].str.upper().str.replace(r"[-_]", "", regex=True)
    )

    # --- Decide alignment path ---
    if n_spectra == n_labels_total:
        # A) Same length -> spectra is full set; build row mask then apply to both
        mask_labeled = labels_df_full["label"].str.strip() != ""
        if args.classes == "2":
            mask_binary = labels_df_full["norm_nosep"].isin(BINARY_KEEP)
            mask_keep = mask_labeled & mask_binary
        else:
            mask_keep = mask_labeled

        labels_df = labels_df_full.loc[mask_keep].copy()
        X_labeled = X_full[mask_keep.values]
    else:
        # B) Spectra already corresponds to a labeled subset; filter labels to match that subset count
        labels_df = labels_df_full[labels_df_full["label"].str.strip() != ""].copy()
        if args.classes == "2":
            labels_df = labels_df[labels_df["norm_nosep"].isin(BINARY_KEEP)].copy()

        if len(labels_df) != n_spectra:
            print(f"Row mismatch after filtering: labels={len(labels_df)}, spectra={n_spectra}.")
            print("For binary runs, labels are filtered to AGN/HM/LM only.")
            print("If counts still differ, your spectra file was generated for a different subset/order. "
                  "Without IDs to join on, we cannot realign rows.")
            return
        X_labeled = X_full

    # 3) Map to class indices; drop unknowns (if any) ----------------------------------------------------
    labels_df["norm_sep"] = labels_df["label"].map(normalize_label_keep_seps)
    labels_df["y_idx"] = labels_df["norm_sep"].map(LABEL_MAP)
    labels_df = labels_df.dropna(subset=["y_idx"]).copy()
    labels_df["y_idx"] = labels_df["y_idx"].astype(int)

    y_all = labels_df["y_idx"].to_numpy()
    X_all = X_labeled

    if len(y_all) == 0:
        print("No labeled rows remain after mapping. Exiting.")
        return

    # Debug: distribution
    vals, cnts = np.unique(y_all, return_counts=True)
    dist = {CLASS_NAMES[int(v)]: int(c) for v, c in zip(vals, cnts)}
    print(f"Post-filter labeled distribution: {dist}")

    # 4) Select class mode ----------------------------------------------------
    if args.classes == "2":
        # Binary = AGN(0) vs HM/LM(1). (YSO/CV/NS/HMXB/LMXB/NS_BIN are excluded upstream)
        # y_all should already be only {0,1}; the mask below is just a safety.
        mask = np.isin(y_all, [0, 1])
        print(f"Keeping only classes 0&1 (AGN vs HM/LM): {int(mask.sum())} of {len(y_all)}")
        X, y = X_all[mask], y_all[mask]
        num_classes = 2
        target_names = ["AGN[0]", "HM/LM[1]"]
    else:
        X, y = X_all, y_all
        num_classes = 4
        target_names = ["AGN[0]", "HM/LM[1]", "YSO/CV[2]", "NS/HMXB/LMXB/NS_BIN[3]"]

    if len(y) == 0:
        print("No data to train on after class filtering. Exiting.")
        return
    if len(np.unique(y)) < num_classes:
        print(f"Only found classes {sorted(np.unique(y).tolist())} after filtering, "
              f"but num_classes={num_classes}. Exiting.")
        return

    print(f"Training on {len(y)} labeled spectra, each with {X.shape[1]} bins, across {num_classes} classes.")

    # 5) Split ----------------------------------------------------
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=args.val_split,
                                          random_state=42, stratify=y)
                        
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
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights_t.to(device))
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # 8) Train + checkpoints --------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    best = {"acc": -1.0, "state": None}
    tag = f"{args.classes}cls_{len(y)}ex_{X.shape[1]}bins"

    for epoch in range(1, args.epochs + 1):
        tr_loss_epoch = train_one_epoch(model, train_loader, criterion, optim, device)

        # Measure TRAIN & VAL on fixed loaders (no sampler)
        tr_loss_eval, tr_acc_eval = evaluate(model, train_eval_loader, criterion, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch:03d} | "
              f"train_loss={tr_loss_epoch:.4f} | train_acc={tr_acc_eval:.4f} | "
              f"val_loss={va_loss:.4f} | val_acc={va_acc:.4f}")

        # Save BEST checkpoint by val_acc
        if va_acc > best["acc"]:
            best = {"acc": va_acc, "state": {k: v.cpu() for k, v in model.state_dict().items()}}
            best_epoch = epoch
            best_ckpt_path = os.path.join(args.out_dir, f"best_{tag}_val{va_acc:.4f}.pt")
            torch.save({
                "state_dict": best["state"],
                "num_classes": num_classes,
                "class_names": CLASS_NAMES,
                "label_map": LABEL_MAP,
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
        "label_map": LABEL_MAP,
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
