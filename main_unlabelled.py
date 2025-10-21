"""
Semi-Supervised Training with Pseudo-Labels

This script trains a spectral classifier using both labelled and unlabelled data.
Uses dynamic pseudo-labeling with curriculum learning (alpha schedule).
"""

import argparse, json
import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pathlib import Path

from utils.model_utils import CNN1D, FocalLoss, train_one_epoch, train_one_epoch_semisupervised, evaluate
from utils.data_utils import (load_data_with_unlabelled, filter_and_remap, make_tensors, 
                               oversample_minority_classes, create_test_set, 
                               create_test_set_unlabelled, save_test_set, load_test_set)
from utils.semisup_utils import (alpha_schedule, alpha_schedule_immediate, 
                                 get_confidence_thresholds, 
                                 compute_pseudo_label_stats, save_pseudo_labels)
from training_visualizer import TrainingVisualizer, should_visualize

torch.manual_seed(42); np.random.seed(42)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)


# Human-readable names (match your grouping)
CLASS_NAMES = {0: "AGN", 1: "HM/LM/YSO", 2: "CV", 3: "NS/HMXB/LMXB/NS_BIN"}



def parse_args():
    ap = argparse.ArgumentParser(description="Train spectra classifier with semi-supervised learning.")
    
    # Basic training params
    ap.add_argument("--classes", choices=["2", "4"], default="4",
                    help="'2' = AGN vs HM/LM/YSO, '4' = 4-class")
    ap.add_argument("--epochs", type=int, default=200,
                    help="Total training epochs (default: 200 for semi-supervised)")
    ap.add_argument("--batch_size", type=int, default=64,
                    help="Batch size for labelled data")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    
    # Data selection
    ap.add_argument("--data", choices=["Brightmos", "Brightpn"], default="Brightpn",
                    help="Dataset: 'Brightpn' (~60k unlabelled) or 'Brightmos' (~27k unlabelled)")
    ap.add_argument("--data_dir", default="data",
                    help="Directory containing data files")
    
    # Semi-supervised params
    ap.add_argument("--unlabelled_batch_size", type=int, default=128,
                    help="Batch size for unlabelled data (can be larger)")
    ap.add_argument("--alpha_warmup", type=int, default=5,
                    help="Epochs before enabling unsupervised loss (default: 10)")
    ap.add_argument("--alpha_max", type=float, default=0.6,
                    help="Maximum alpha for unsupervised loss weight (default: 0.8)")
    ap.add_argument("--alpha_schedule", choices=["standard", "immediate"], default="immediate",
                    help="'standard' = warmup then ramp, 'immediate' = start from epoch 1 (for fast learners)")
    
    # Confidence thresholds (class-specific for 4-class)
    ap.add_argument("--conf_threshold_agn", type=float, default=0.8,
                    help="Confidence threshold for AGN pseudo-labels")
    ap.add_argument("--conf_threshold_hmlm", type=float, default=0.9,
                    help="Confidence threshold for HM/LM/YSO pseudo-labels")
    ap.add_argument("--conf_threshold_cv", type=float, default=0.5,
                    help="Confidence threshold for CV pseudo-labels")
    ap.add_argument("--conf_threshold_ns", type=float, default=0.5,
                    help="Confidence threshold for NS pseudo-labels")
    
    # Test set params
    ap.add_argument("--test_samples_labelled", type=int, default=10,
                    help="Total number of labelled test samples (stratified across all classes, default: 10)")
    ap.add_argument("--test_samples_unlabelled", type=int, default=10,
                    help="Total number of unlabelled test samples (default: 10)")
    ap.add_argument("--test_set_dir", default="test_sets",
                    help="Directory to save/load test sets")
    ap.add_argument("--load_test_set", action="store_true",
                    help="Load existing test set instead of creating new one")
    
    # Other params (from main.py)
    ap.add_argument("--oversample_target", type=float, default=0.7,
                    help="Target ratio for minority oversampling (4-class only)")
    ap.add_argument("--focal_gamma", type=float, default=2.0,
                    help="Focal loss gamma parameter")
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", default="checkpoints",
                    help="Directory for model checkpoints")
    ap.add_argument("--save_preds", default=None,
                    help="Path to save test predictions JSON")
    
    # Visualization
    ap.add_argument("--visualize_training", action="store_true",
                    help="Enable real-time latent space visualization")
    ap.add_argument("--viz_every", type=int, default=5,
                    help="Visualize every N epochs")
    ap.add_argument("--viz_method", choices=["umap", "tsne"], default="tsne")
    
    return ap.parse_args()


def norm_per_sample(A):
    """Per-spectrum normalization (zero-mean, unit-std per row)."""
    mu = A.mean(axis=1, keepdims=True)
    sigma = A.std(axis=1, keepdims=True) + 1e-8
    return (A - mu) / sigma


def main():
    args = parse_args()
    device = torch.device(args.device)
    
    # Create output directories
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.test_set_dir, exist_ok=True)
    
    # Data file paths
    data_file = f"{args.data_dir}/{args.data}_id_normspec_counts_label.txt"
    test_set_labelled_path = f"{args.test_set_dir}/test_labelled_{args.data}_{args.classes}cls.json"
    test_set_unlabelled_path = f"{args.test_set_dir}/test_unlabelled_{args.data}.json"
    
    print("="*80)
    print(f"SEMI-SUPERVISED TRAINING: {args.classes}-class with {args.data} data")
    print("="*80)
    
    # ========================================================================
    # 1. Load data (labelled + unlabelled)
    # ========================================================================
    print(f"\n📂 Loading data from {data_file}...")
    X_lab_all, y_lab_all, src_ids_lab, counts_lab, X_unlab_all, src_ids_unlab, counts_unlab = load_data_with_unlabelled(data_file)
    
    # Filter and remap labels based on class mode
    if args.classes == "2":
        mode = "12"
        num_classes = 2
    else:
        mode = "all4"
        num_classes = 4
    
    y_lab_filtered, printable_map, target_names = filter_and_remap(y_lab_all, mode)
    
    # Apply filtering to X_labelled
    if args.classes == "2":
        mask = np.isin(y_lab_all, [0, 1])
        X_lab_filtered = X_lab_all[mask]
        src_ids_lab_filtered = src_ids_lab[mask]
        counts_lab_filtered = counts_lab[mask]
    else:
        X_lab_filtered = X_lab_all
        src_ids_lab_filtered = src_ids_lab
        counts_lab_filtered = counts_lab
        y_lab_filtered = y_lab_all
    
    print(f"\n📊 Data summary:")
    print(f"  Labelled: {len(X_lab_filtered)} samples")
    print(f"  Unlabelled: {len(X_unlab_all)} samples")
    print(f"  Classes: {num_classes}")
    
    # Class distribution
    vals, cnts = np.unique(y_lab_filtered, return_counts=True)
    dist = {CLASS_NAMES[int(v)]: int(c) for v, c in zip(vals, cnts)}
    print(f"  Label distribution: {dist}")
    
    # ========================================================================
    # 2. Create or load test sets
    # ========================================================================
    if args.load_test_set and os.path.exists(test_set_labelled_path):
        print(f"\n📥 Loading existing test sets...")
        X_test_lab, y_test_lab, src_ids_test_lab = load_test_set(test_set_labelled_path)
        X_test_unlab, _, src_ids_test_unlab = load_test_set(test_set_unlabelled_path)
        
        # Remove test samples from training data (match by source IDs)
        test_ids_set = set(src_ids_test_lab).union(set(src_ids_test_unlab))
        
        # Filter labelled data
        mask_lab = ~np.isin(src_ids_lab_filtered, list(test_ids_set))
        X_lab_filtered = X_lab_filtered[mask_lab]
        y_lab_filtered = y_lab_filtered[mask_lab]
        src_ids_lab_filtered = src_ids_lab_filtered[mask_lab]
        counts_lab_filtered = counts_lab_filtered[mask_lab]
        
        # Filter unlabelled data
        mask_unlab = ~np.isin(src_ids_unlab, list(test_ids_set))
        X_unlab_all = X_unlab_all[mask_unlab]
        src_ids_unlab = src_ids_unlab[mask_unlab]
        
    else:
        print(f"\n🎲 Creating new test sets...")
        
        # Create test set from labelled data (stratified, 10 total samples)
        (X_lab_filtered, y_lab_filtered, src_ids_lab_filtered, counts_lab_filtered,
         X_test_lab, y_test_lab, src_ids_test_lab, counts_test_lab) = create_test_set(
            X_lab_filtered, y_lab_filtered, src_ids_lab_filtered, counts_lab_filtered,
            n_samples_total=args.test_samples_labelled, random_state=42
        )
        
        # Create test set from unlabelled data (random)
        (X_unlab_all, src_ids_unlab, 
         X_test_unlab, src_ids_test_unlab) = create_test_set_unlabelled(
            X_unlab_all, src_ids_unlab, 
            n_samples=args.test_samples_unlabelled, random_state=42
        )
        
        # Save test sets
        save_test_set(X_test_lab, y_test_lab, src_ids_test_lab, 
                     test_set_labelled_path, CLASS_NAMES)
        save_test_set(X_test_unlab, None, src_ids_test_unlab, 
                     test_set_unlabelled_path, None)
    
    print(f"\n✅ Test sets ready:")
    print(f"  Labelled test: {len(X_test_lab)} samples")
    print(f"  Unlabelled test: {len(X_test_unlab)} samples")
    print(f"  Remaining for training:")
    print(f"    Labelled: {len(X_lab_filtered)} samples")
    print(f"    Unlabelled: {len(X_unlab_all)} samples")
    
    # ========================================================================
    # 3. Split labelled data into train/val
    # ========================================================================
    Xtr, Xva, ytr, yva, src_ids_tr, src_ids_va, counts_tr, counts_va = train_test_split(
        X_lab_filtered, y_lab_filtered, src_ids_lab_filtered, counts_lab_filtered,
        test_size=args.val_split, random_state=42, stratify=y_lab_filtered
    )
    
    # Apply oversampling for 4-class mode
    if args.classes == "4":
        print(f"\n⚖️  Oversampling minority classes...")
        print(f"  Original: {dict(zip(*np.unique(ytr, return_counts=True)))}")
        
        from collections import Counter
        class_counts = Counter(ytr)
        target_count = int(max(class_counts.values()) * args.oversample_target)
        Xtr, ytr = oversample_minority_classes(
            Xtr, ytr, target_count=target_count, augment=True, noise_std=0.005
        )
        
        print(f"  After: {dict(zip(*np.unique(ytr, return_counts=True)))}")
    
    # Normalize spectra
    Xtr = norm_per_sample(Xtr)
    Xva = norm_per_sample(Xva)
    X_unlab_all = norm_per_sample(X_unlab_all)
    X_test_lab = norm_per_sample(X_test_lab)
    X_test_unlab = norm_per_sample(X_test_unlab)
    
    # ========================================================================
    # 4. Create dataloaders
    # ========================================================================
    # Labelled data tensors
    Xtr_t, ytr_t = make_tensors(Xtr, ytr)
    Xva_t, yva_t = make_tensors(Xva, yva)
    
    # Unlabelled data tensors (dummy labels)
    X_unlab_t, _ = make_tensors(X_unlab_all, np.zeros(len(X_unlab_all), dtype=np.int64))
    
    # Compute class weights
    class_counts = np.bincount(ytr, minlength=num_classes)
    class_weights = (class_counts.sum() / (class_counts + 1e-8)).astype(np.float32)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32)
    
    # Labelled dataloader with balanced sampling
    train_ds = TensorDataset(Xtr_t, ytr_t)
    ex_weights = class_weights_t[ytr_t].cpu().double()
    sampler = WeightedRandomSampler(
        weights=ex_weights, num_samples=len(ex_weights), replacement=True
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler, 
        shuffle=False, num_workers=args.num_workers
    )
    
    # Evaluation loaders (no sampling)
    train_eval_loader = DataLoader(
        TensorDataset(Xtr_t, ytr_t), batch_size=args.batch_size, 
        shuffle=False, num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        TensorDataset(Xva_t, yva_t), batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers
    )
    
    # Unlabelled dataloader (larger batches OK)
    unlabelled_loader = DataLoader(
        TensorDataset(X_unlab_t, X_unlab_t[:, 0, 0]),  # Dummy target
        batch_size=args.unlabelled_batch_size, shuffle=True, num_workers=args.num_workers
    )
    
    print(f"\n📦 Dataloaders ready:")
    print(f"  Train batches: {len(train_loader)} × {args.batch_size}")
    print(f"  Unlabelled batches: {len(unlabelled_loader)} × {args.unlabelled_batch_size}")
    print(f"  Val batches: {len(val_loader)} × {args.batch_size}")
    
    # ========================================================================
    # 5. Model, optimizer, criterion
    # ========================================================================
    model = CNN1D(num_classes=num_classes).to(device)
    criterion = FocalLoss(alpha=class_weights_t.to(device), gamma=args.focal_gamma)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )
    
    # Confidence thresholds
    if args.classes == "2":
        conf_thresholds = {
            0: args.conf_threshold_agn,
            1: args.conf_threshold_hmlm,
        }
    else:
        conf_thresholds = {
            0: args.conf_threshold_agn,
            1: args.conf_threshold_hmlm,
            2: args.conf_threshold_cv,
            3: args.conf_threshold_ns,
        }
    
    print(f"\n🎯 Confidence thresholds: {conf_thresholds}")
    
    # Visualizer (optional)
    visualizer = None
    if args.visualize_training:
        visualizer = TrainingVisualizer(
            num_classes=num_classes, run_name=None, method=args.viz_method,
            enable_gradcam=True,  # Enable GradCAM to visualize receptive fields
            train_metadata={'src_ids': src_ids_tr, 'counts': counts_tr},
            val_metadata={'src_ids': src_ids_va, 'counts': counts_va}
        )
    
    # ========================================================================
    # 6. Training loop with semi-supervised learning
    # ========================================================================
    best = {"acc": -1.0, "state": None, "epoch": 0, "loss": float('inf')}
    tag = f"semisup_{args.classes}cls_{args.data}"
    
    print("\n" + "="*80)
    print("TRAINING START")
    print("="*80)
    
    # Print alpha schedule info
    if args.alpha_schedule == "immediate":
        print(f"🔥 Using IMMEDIATE alpha schedule:")
        print(f"   Epoch 1-{args.alpha_warmup}: alpha ramps 0.1 → 0.3")
        print(f"   Epoch {args.alpha_warmup+1}-{args.epochs}: alpha ramps 0.3 → {args.alpha_max}")
    else:
        print(f"📚 Using STANDARD alpha schedule:")
        print(f"   Epoch 1-{args.alpha_warmup}: alpha = 0.0 (supervised only)")
        print(f"   Epoch {args.alpha_warmup+1}-{args.epochs}: alpha ramps 0.0 → {args.alpha_max}")
    print()
    
    for epoch in range(1, args.epochs + 1):
        # Compute alpha for this epoch
        if args.alpha_schedule == "immediate":
            alpha = alpha_schedule_immediate(epoch, args.epochs, args.alpha_warmup, args.alpha_max)
        else:
            alpha = alpha_schedule(epoch, args.epochs, args.alpha_warmup, args.alpha_max)
        
        # Train one epoch
        train_stats = train_one_epoch_semisupervised(
            model, train_loader, unlabelled_loader, criterion, optim, device,
            alpha, conf_thresholds
        )
        
        # Evaluate
        tr_loss_eval, tr_acc_eval = evaluate(model, train_eval_loader, criterion, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        
        # Print progress
        print(f"Epoch {epoch:03d} | alpha={alpha:.3f} | "
              f"sup_loss={train_stats['loss_sup']:.4f} | unsup_loss={train_stats['loss_unsup']:.4f} | "
              f"pseudo_accept={train_stats['pseudo_accept_rate']:.2%} | "
              f"train_acc={tr_acc_eval:.4f} | val_acc={va_acc:.4f}")
        
        # Print pseudo-label class distribution
        if alpha > 0 and train_stats['pseudo_class_distribution']:
            dist_str = ", ".join([f"{CLASS_NAMES[k]}:{v}" for k, v in 
                                 sorted(train_stats['pseudo_class_distribution'].items())])
            print(f"  ↳ Pseudo-labels: {dist_str}")
        
        # Learning rate scheduling
        old_lr = optim.param_groups[0]['lr']
        scheduler.step(va_loss)
        new_lr = optim.param_groups[0]['lr']
        if new_lr < old_lr:
            print(f"  ↳ Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")
        
        # Visualization
        if visualizer and should_visualize(epoch, args.epochs, args.viz_every):
            visualizer.visualize_epoch(
                model, train_eval_loader, val_loader, device,
                epoch, tr_acc_eval, va_acc, tr_loss_eval, va_loss
            )
        
        # Save best checkpoint
        if va_acc > best["acc"]:
            best = {
                "acc": va_acc, "loss": va_loss, "epoch": epoch,
                "state": {k: v.cpu().clone() for k, v in model.state_dict().items()}
            }
            best_ckpt_path = os.path.join(args.out_dir, f"best_{tag}_epoch{epoch:03d}_val{va_acc:.4f}.pt")
            torch.save({
                "state_dict": best["state"],
                "num_classes": num_classes,
                "class_names": CLASS_NAMES,
                "args": vars(args),
                "best_epoch": epoch,
            }, best_ckpt_path)
            print(f"  ↳ ✅ Saved BEST checkpoint: {best_ckpt_path}")
    
    # ========================================================================
    # 7. Evaluate both BEST and LAST checkpoints
    # ========================================================================
    
    # Save last checkpoint state before loading best
    last_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    last_ckpt_path = os.path.join(args.out_dir, f"last_{tag}_epoch{epoch:03d}_val{va_acc:.4f}.pt")
    torch.save({
        "state_dict": last_state,
        "num_classes": num_classes,
        "class_names": CLASS_NAMES,
        "args": vars(args),
        "last_epoch": epoch,
    }, last_ckpt_path)
    print(f"\n💾 Saved LAST checkpoint: {last_ckpt_path}")
    
    # Prepare test dataloaders
    X_test_lab_t, y_test_lab_t = make_tensors(X_test_lab, y_test_lab)
    test_lab_loader = DataLoader(
        TensorDataset(X_test_lab_t, y_test_lab_t), batch_size=args.batch_size, shuffle=False
    )
    
    X_test_unlab_t, _ = make_tensors(X_test_unlab, np.zeros(len(X_test_unlab), dtype=np.int64))
    test_unlab_loader = DataLoader(
        TensorDataset(X_test_unlab_t, X_test_unlab_t[:, 0, 0]), 
        batch_size=args.batch_size, shuffle=False
    )
    
    # Function to evaluate a checkpoint on all datasets
    def evaluate_checkpoint(model_state, checkpoint_name, checkpoint_epoch):
        """Evaluate a checkpoint on validation, labelled test, and unlabelled test sets."""
        model.load_state_dict(model_state)
        model.to(device)
        model.eval()
        
        print("\n" + "="*80)
        print(f"=== {checkpoint_name} (Epoch {checkpoint_epoch}) ===")
        print("="*80)
        
        # Validation set
        with torch.no_grad():
            y_pred_val = []
            for xb, _ in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                y_pred_val.append(torch.argmax(logits, dim=1).cpu().numpy())
            y_pred_val = np.concatenate(y_pred_val)
        
        val_acc_final = accuracy_score(yva, y_pred_val)
        print(f"\n📊 Validation Set:")
        print(f"Accuracy: {val_acc_final:.4f}")
        print("\nClassification report:")
        print(classification_report(yva, y_pred_val, target_names=target_names, digits=4))
        print("Confusion matrix:\n", confusion_matrix(yva, y_pred_val))
        
        # Labelled test set
        with torch.no_grad():
            y_pred_test_lab = []
            y_prob_test_lab = []
            for xb, _ in test_lab_loader:
                xb = xb.to(device)
                logits = model(xb)
                probs = torch.softmax(logits, dim=1)
                y_pred_test_lab.append(torch.argmax(logits, dim=1).cpu().numpy())
                y_prob_test_lab.append(probs.cpu().numpy())
            y_pred_test_lab = np.concatenate(y_pred_test_lab)
            y_prob_test_lab = np.vstack(y_prob_test_lab)
        
        test_acc = accuracy_score(y_test_lab, y_pred_test_lab)
        print(f"\n📊 Held-Out Labelled Test Set:")
        print(f"Accuracy: {test_acc:.4f}")
        print("\nClassification report:")
        print(classification_report(y_test_lab, y_pred_test_lab, target_names=target_names, digits=4))
        print("Confusion matrix:\n", confusion_matrix(y_test_lab, y_pred_test_lab))
        
        # Unlabelled test set
        with torch.no_grad():
            y_pred_test_unlab = []
            y_prob_test_unlab = []
            for xb, _ in test_unlab_loader:
                xb = xb.to(device)
                logits = model(xb)
                probs = torch.softmax(logits, dim=1)
                y_pred_test_unlab.append(torch.argmax(logits, dim=1).cpu().numpy())
                y_prob_test_unlab.append(probs.cpu().numpy())
            y_pred_test_unlab = np.concatenate(y_pred_test_unlab)
            y_prob_test_unlab = np.vstack(y_prob_test_unlab)
        
        avg_conf = np.mean([y_prob_test_unlab[i, pred] for i, pred in enumerate(y_pred_test_unlab)])
        print(f"\n📊 Held-Out Unlabelled Test Set:")
        print(f"Average confidence: {avg_conf:.3f}")
        print("Predictions:")
        for i, (src_id, pred, prob) in enumerate(zip(src_ids_test_unlab, y_pred_test_unlab, y_prob_test_unlab)):
            conf = prob[pred]
            print(f"{i+1:2d}. {src_id:20s} → {CLASS_NAMES[pred]:20s} (conf: {conf:.3f})")
        
        return {
            'val_acc': val_acc_final,
            'val_preds': y_pred_val,
            'test_acc': test_acc,
            'test_preds': y_pred_test_lab,
            'test_probs': y_prob_test_lab,
            'unlab_preds': y_pred_test_unlab,
            'unlab_probs': y_prob_test_unlab,
            'avg_unlab_conf': avg_conf,
        }
    
    # Evaluate BOTH checkpoints
    print("\n" + "="*80)
    print("EVALUATING BOTH CHECKPOINTS")
    print("="*80)
    
    best_results = evaluate_checkpoint(best["state"], "BEST MODEL", best['epoch'])
    last_results = evaluate_checkpoint(last_state, "LAST MODEL", epoch)
    
    # COMPARISON SUMMARY
    print("\n" + "="*80)
    print("=== CHECKPOINT COMPARISON SUMMARY ===")
    print("="*80)
    print(f"\n{'Metric':<35} {'BEST (E{})'.format(best['epoch']):<20} {'LAST (E{})'.format(epoch):<20} {'Winner'}")
    print("-" * 95)
    print(f"{'Validation Accuracy':<35} {best_results['val_acc']:.4f}{' '*16} {last_results['val_acc']:.4f}{' '*16} {'✅ BEST' if best_results['val_acc'] > last_results['val_acc'] else '✅ LAST' if last_results['val_acc'] > best_results['val_acc'] else '🤝 TIE'}")
    print(f"{'Labelled Test Accuracy':<35} {best_results['test_acc']:.4f}{' '*16} {last_results['test_acc']:.4f}{' '*16} {'✅ BEST' if best_results['test_acc'] > last_results['test_acc'] else '✅ LAST' if last_results['test_acc'] > best_results['test_acc'] else '🤝 TIE'}")
    print(f"{'Unlabelled Avg Confidence':<35} {best_results['avg_unlab_conf']:.4f}{' '*16} {last_results['avg_unlab_conf']:.4f}{' '*16} {'✅ BEST' if best_results['avg_unlab_conf'] > last_results['avg_unlab_conf'] else '✅ LAST' if last_results['avg_unlab_conf'] > best_results['avg_unlab_conf'] else '🤝 TIE'}")
    
    # Check agreement on unlabelled test set
    agreement = (best_results['unlab_preds'] == last_results['unlab_preds']).sum()
    print(f"\n{'Unlabelled Test Agreement':<35} {agreement}/{len(src_ids_test_unlab)} predictions match")
    
    if agreement < len(src_ids_test_unlab):
        print("\n🔍 Disagreements on unlabelled test set:")
        for i in range(len(src_ids_test_unlab)):
            if best_results['unlab_preds'][i] != last_results['unlab_preds'][i]:
                best_pred = best_results['unlab_preds'][i]
                last_pred = last_results['unlab_preds'][i]
                best_conf = best_results['unlab_probs'][i, best_pred]
                last_conf = last_results['unlab_probs'][i, last_pred]
                print(f"  {src_ids_test_unlab[i]:20s}: BEST={CLASS_NAMES[best_pred]:15s} (conf={best_conf:.3f}), "
                      f"LAST={CLASS_NAMES[last_pred]:15s} (conf={last_conf:.3f})")
    
    # Recommendation
    print("\n💡 Recommendation:")
    if best_results['val_acc'] >= last_results['val_acc'] and best_results['test_acc'] >= last_results['test_acc']:
        print("   Use BEST checkpoint - it generalizes better and hasn't overfit to pseudo-labels.")
    elif last_results['test_acc'] > best_results['test_acc'] and last_results['avg_unlab_conf'] > best_results['avg_unlab_conf']:
        print("   Consider LAST checkpoint - it may have learned useful patterns from unlabelled data.")
    else:
        print("   BEST checkpoint is safer, but inspect unlabelled predictions carefully.")
    
    # Save predictions to JSON (both checkpoints)
    if args.save_preds:
        predictions = {
            'best_checkpoint': {
                'epoch': best['epoch'],
                'val_acc': float(best['acc']),
                'labelled_test': [
                    {
                        'source_id': str(src_ids_test_lab[i]),
                        'true_label': int(y_test_lab[i]),
                        'true_label_name': CLASS_NAMES[int(y_test_lab[i])],
                        'predicted': int(best_results['test_preds'][i]),
                        'predicted_name': CLASS_NAMES[int(best_results['test_preds'][i])],
                        'confidence': float(best_results['test_probs'][i, best_results['test_preds'][i]]),
                        'correct': bool(y_test_lab[i] == best_results['test_preds'][i]),
                    }
                    for i in range(len(y_test_lab))
                ],
                'unlabelled_test': [
                    {
                        'source_id': str(src_ids_test_unlab[i]),
                        'predicted': int(best_results['unlab_preds'][i]),
                        'predicted_name': CLASS_NAMES[int(best_results['unlab_preds'][i])],
                        'confidence': float(best_results['unlab_probs'][i, best_results['unlab_preds'][i]]),
                        'probabilities': {CLASS_NAMES[j]: float(best_results['unlab_probs'][i, j]) 
                                         for j in range(num_classes)},
                        'manual_label': None,
                    }
                    for i in range(len(best_results['unlab_preds']))
                ],
                'test_acc_labelled': float(best_results['test_acc']),
                'avg_unlab_conf': float(best_results['avg_unlab_conf']),
            },
            'last_checkpoint': {
                'epoch': epoch,
                'val_acc': float(last_results['val_acc']),
                'labelled_test': [
                    {
                        'source_id': str(src_ids_test_lab[i]),
                        'true_label': int(y_test_lab[i]),
                        'true_label_name': CLASS_NAMES[int(y_test_lab[i])],
                        'predicted': int(last_results['test_preds'][i]),
                        'predicted_name': CLASS_NAMES[int(last_results['test_preds'][i])],
                        'confidence': float(last_results['test_probs'][i, last_results['test_preds'][i]]),
                        'correct': bool(y_test_lab[i] == last_results['test_preds'][i]),
                    }
                    for i in range(len(y_test_lab))
                ],
                'unlabelled_test': [
                    {
                        'source_id': str(src_ids_test_unlab[i]),
                        'predicted': int(last_results['unlab_preds'][i]),
                        'predicted_name': CLASS_NAMES[int(last_results['unlab_preds'][i])],
                        'confidence': float(last_results['unlab_probs'][i, last_results['unlab_preds'][i]]),
                        'probabilities': {CLASS_NAMES[j]: float(last_results['unlab_probs'][i, j]) 
                                         for j in range(num_classes)},
                        'manual_label': None,
                    }
                    for i in range(len(last_results['unlab_preds']))
                ],
                'test_acc_labelled': float(last_results['test_acc']),
                'avg_unlab_conf': float(last_results['avg_unlab_conf']),
            },
            'model_info': {
                'classes': args.classes,
                'data': args.data,
                'alpha_schedule': args.alpha_schedule,
                'alpha_max': args.alpha_max,
                'alpha_warmup': args.alpha_warmup,
                'conf_thresholds': {CLASS_NAMES[k]: v for k, v in conf_thresholds.items()},
            }
        }
        
        with open(args.save_preds, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        print(f"\n💾 Saved predictions (BEST and LAST) to: {args.save_preds}")
    
    # Animation script
    if visualizer:
        visualizer.create_animation_script()
        print(f"\n📽️  To create training animation:")
        print(f"   cd {visualizer.run_dir} && ./create_animation.sh")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

