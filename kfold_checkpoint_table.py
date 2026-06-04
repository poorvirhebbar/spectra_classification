"""
Print a table mapping every k-fold training to its fold checkpoints.

Scans checkpoints_kfold/ for cv_summary_*.json files (written by main_kfold.py and
main_unlabelled_kfold.py) and lists, per task, the 5 fold checkpoints with their
val/test accuracies. Expected total: 8 tasks x 5 folds = 40 checkpoints.

Usage:  python kfold_checkpoint_table.py [--root checkpoints_kfold]
"""
import argparse, glob, json, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="checkpoints_kfold")
    args = ap.parse_args()

    summaries = sorted(glob.glob(os.path.join(args.root, "**", "cv_summary_*.json"), recursive=True))
    if not summaries:
        print(f"No cv_summary_*.json found under {args.root}/ yet.")
        return

    rows = []
    for sp in summaries:
        with open(sp) as f:
            s = json.load(f)
        task = f"{s['data']}_{s['classes']}cls_{s['mode']}"
        ckpts = s.get("fold_checkpoints", [])
        vacc = s.get("val_accuracies", [])
        tacc = s.get("test_accuracies", [])
        best = s.get("best_fold", -1)
        for i, ck in enumerate(ckpts):
            rows.append({
                "task": task,
                "fold": i,
                "ckpt": os.path.relpath(ck) if os.path.exists(ck) else ck,
                "val": vacc[i] if i < len(vacc) else float("nan"),
                "test": tacc[i] if i < len(tacc) else float("nan"),
                "best": "  <-- best" if i == best else "",
            })

    print(f"\n{'task':<28} {'fold':>4} {'val_acc':>8} {'test_acc':>8}  checkpoint")
    print("-" * 110)
    last = None
    for r in rows:
        if r["task"] != last:
            if last is not None:
                print()
            last = r["task"]
        print(f"{r['task']:<28} {r['fold']:>4} {r['val']:>8.4f} {r['test']:>8.4f}  "
              f"{os.path.basename(r['ckpt'])}{r['best']}")

    print("-" * 110)
    print(f"Total: {len(rows)} checkpoints across {len(summaries)} tasks "
          f"(expected 40 across 8 tasks when all runs finish).")


if __name__ == "__main__":
    main()
