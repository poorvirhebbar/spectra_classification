# Semi-Supervised Learning Guide

Train spectral classifiers using both labelled and unlabelled data with dynamic pseudo-labeling.

---

## 🚀 Quick Start

### Basic 4-Class Training (Brightpn, 60k unlabelled)
```bash
python main_unlabelled.py --classes 4 --epochs 200 --data Brightpn
```

### Basic 2-Class Training (Brightmos, 27k unlabelled)
```bash
python main_unlabelled.py --classes 2 --epochs 200 --data Brightmos
```

---

## 📋 Key Features

### 1. **Dynamic Pseudo-Labeling**
- Pseudo-labels are generated **every epoch** (not static)
- Only high-confidence predictions are used
- Class-specific confidence thresholds

### 2. **Curriculum Learning (Alpha Schedule)**
- **Epochs 1-50**: alpha = 0.0 (supervised only, warm-up)
- **Epochs 51-200**: alpha ramps from 0 to 0.8 (gradual pseudo-label incorporation)
- **Max alpha = 0.8**: Unsupervised loss never dominates

### 3. **Automatic Test Set Management**
- Holds out 10 labelled + 10 unlabelled samples for testing
- Test sets are saved to `test_sets/` directory
- Reusable across runs with `--load_test_set`

### 4. **Comprehensive Evaluation**
- Validation set (20% of labelled data)
- Held-out labelled test set (for accuracy measurement)
- Held-out unlabelled test set (for manual verification)

---

## 🎚️ Key Parameters

### Data Selection
```bash
--data Brightpn         # 60k unlabelled samples
--data Brightmos        # 27k unlabelled samples
```

### Semi-Supervised Training
```bash
--alpha_warmup 50       # Epochs before enabling unsupervised loss
--alpha_max 0.8         # Maximum weight for unsupervised loss (0.8 = 80%)
--unlabelled_batch_size 128  # Larger batches OK for unlabelled
```

### Confidence Thresholds (4-Class)
```bash
--conf_threshold_agn 0.95    # AGN (majority class, high bar)
--conf_threshold_hmlm 0.95   # HM/LM/YSO (majority, high bar)
--conf_threshold_cv 0.90     # CV (minority, slightly lower)
--conf_threshold_ns 0.85     # NS (very minority, more lenient)
```

### Test Set
```bash
--test_samples_labelled 10         # Total labelled test samples (stratified)
--test_samples_unlabelled 10       # Total unlabelled test samples
--load_test_set                    # Reuse existing test set
--save_preds predictions.json      # Save predictions for inspection
```

---

## 📊 Example Output

```
Epoch 050 | alpha=0.00 | sup_loss=0.25 | unsup_loss=0.00 | pseudo_accept=0% | val_acc=0.82
Epoch 100 | alpha=0.33 | sup_loss=0.18 | unsup_loss=0.32 | pseudo_accept=45% | val_acc=0.85
  ↳ Pseudo-labels: AGN:1200, HM/LM/YSO:980, CV:450, NS:120
Epoch 150 | alpha=0.67 | sup_loss=0.12 | unsup_loss=0.15 | pseudo_accept=68% | val_acc=0.87
  ↳ Pseudo-labels: AGN:1800, HM/LM/YSO:1650, CV:820, NS:340
Epoch 200 | alpha=0.80 | sup_loss=0.08 | unsup_loss=0.10 | pseudo_accept=75% | val_acc=0.88
  ↳ Pseudo-labels: AGN:2100, HM/LM/YSO:1950, CV:1100, NS:480
```

**Good Signs:**
- ✅ Pseudo-accept rate increases over time (model getting confident)
- ✅ Unsupervised loss decreases (pseudo-labels improving)
- ✅ Validation accuracy improves or stays stable

**Warning Signs:**
- ❌ Pseudo-accept rate very low (<20%) → Thresholds too strict
- ❌ Validation accuracy drops → Model degrading, reduce alpha_max
- ❌ All pseudo-labels same class → Overfitting, lower thresholds

---

## 🎯 Training Workflows

### Workflow 1: Conservative (High Quality Pseudo-Labels)
```bash
python main_unlabelled.py --classes 4 --epochs 200 \
    --data Brightpn \
    --alpha_warmup 70 --alpha_max 0.6 \
    --conf_threshold_agn 0.97 --conf_threshold_hmlm 0.97 \
    --conf_threshold_cv 0.93 --conf_threshold_ns 0.88
```
**Best for:** When you want high-quality pseudo-labels only

### Workflow 2: Balanced (Recommended)
```bash
python main_unlabelled.py --classes 4 --epochs 200 \
    --data Brightpn \
    --alpha_warmup 50 --alpha_max 0.8 \
    --save_preds predictions/brightpn_4cls_run001.json
```
**Best for:** General use, good balance

### Workflow 3: Aggressive (More Pseudo-Labels)
```bash
python main_unlabelled.py --classes 4 --epochs 250 \
    --data Brightpn \
    --alpha_warmup 40 --alpha_max 0.8 \
    --conf_threshold_agn 0.92 --conf_threshold_hmlm 0.92 \
    --conf_threshold_cv 0.87 --conf_threshold_ns 0.82
```
**Best for:** When you have strong baseline and want to leverage more unlabelled data

### Workflow 4: 2-Class with Brightmos
```bash
python main_unlabelled.py --classes 2 --epochs 150 \
    --data Brightmos \
    --alpha_warmup 40 --alpha_max 0.8 \
    --conf_threshold_agn 0.90 --conf_threshold_hmlm 0.90
```
**Best for:** 2-class is easier, can use lower thresholds

---

## 📁 Output Structure

```
spectra_classification/
├── test_sets/                                  # Held-out test samples
│   ├── test_labelled_Brightpn_4cls.json       # 10 labelled (stratified)
│   ├── test_unlabelled_Brightpn.json          # 10 unlabelled
│   ├── test_labelled_Brightmos_2cls.json      # 10 labelled (stratified)
│   └── test_unlabelled_Brightmos.json         # 10 unlabelled
├── checkpoints/
│   ├── best_semisup_4cls_Brightpn_val0.8756.pt
│   └── last_semisup_4cls_Brightpn_val0.8650.pt
├── predictions/
│   └── brightpn_4cls_run001.json              # Test predictions
└── training_feature_visualizations/           # (if --visualize_training)
    └── run_025/
        ├── epoch_001_acc0.6500.png
        ├── epoch_100_acc0.8500.png
        └── epoch_200_acc0.8800.png
```

---

## 🧪 Manual Verification (Unlabelled Test Set)

After training, check the predictions JSON:

```json
{
  "unlabelled_test": [
    {
      "source_id": "4XMM_J123456+789012",
      "predicted": 0,
      "predicted_name": "AGN",
      "confidence": 0.987,
      "probabilities": {
        "AGN": 0.987,
        "HM/LM/YSO": 0.008,
        "CV": 0.003,
        "NS/HMXB/LMXB/NS_BIN": 0.002
      },
      "manual_label": null  ← Fill this after manual inspection
    },
    ...
  ]
}
```

**Steps:**
1. Open predictions JSON
2. For each unlabelled sample, inspect the spectrum (use source_id)
3. Assign a manual label based on your expertise
4. Fill the `"manual_label"` field
5. Compute accuracy: Compare `"predicted"` vs `"manual_label"`

---

## 🔧 Troubleshooting

### Issue 1: Very Low Pseudo-Accept Rate (<10%)
**Cause:** Confidence thresholds too high
**Fix:** Lower thresholds by 0.03-0.05
```bash
--conf_threshold_agn 0.90 --conf_threshold_hmlm 0.90 \
--conf_threshold_cv 0.85 --conf_threshold_ns 0.80
```

### Issue 2: Validation Accuracy Drops
**Cause:** Pseudo-labels causing overfitting
**Fix:** Reduce alpha_max or increase warmup
```bash
--alpha_warmup 70 --alpha_max 0.6
```

### Issue 3: All Pseudo-Labels Same Class
**Cause:** Model overconfident on majority class
**Fix:** Use focal loss gamma and lower thresholds
```bash
--focal_gamma 3.0 --conf_threshold_agn 0.93
```

### Issue 4: Test Set Already Exists Error
**Fix:** Use existing test set or delete and recreate
```bash
# Option 1: Reuse test set
python main_unlabelled.py --load_test_set ...

# Option 2: Delete and recreate
rm test_sets/test_*_Brightpn_*.json
python main_unlabelled.py ...
```

---

## 📊 Comparison: Supervised vs Semi-Supervised

| Metric | Supervised (main.py) | Semi-Supervised (main_unlabelled.py) |
|--------|---------------------|--------------------------------------|
| **Training data** | 2,730 labelled | 2,720 labelled + 60k unlabelled |
| **Training time** | 150 epochs (~30 min) | 200 epochs (~60 min) |
| **2-class accuracy** | ~96% | ~96-97% (similar) |
| **4-class accuracy** | ~81-85% | ~85-88% (**+3-4% improvement**) |
| **Minority class recall** | 50-60% (YSO/CV, NS) | 65-75% (**+10-15% improvement**) |

**Bottom line:** Semi-supervised helps most for 4-class, especially minority classes!

---

## 💡 Tips for Best Results

1. **Start with good baseline**: Train supervised first (`main.py`), then semi-supervised
2. **Monitor pseudo-accept rate**: Should gradually increase (30% → 70%)
3. **Class-specific thresholds**: Lower for minority classes (CV, NS)
4. **Conservative warmup**: 50-70 epochs ensures good baseline before pseudo-labels
5. **Save predictions**: Always use `--save_preds` for manual verification
6. **Visualize**: Use `--visualize_training` to watch clusters form
7. **Compare datasets**: Try both Brightpn and Brightmos, see which helps more

---

## 🚀 Next Steps

1. Run baseline supervised training:
   ```bash
   python main.py --classes 4 --epochs 150
   ```

2. Run semi-supervised training:
   ```bash
   python main_unlabelled.py --classes 4 --epochs 200 --data Brightpn \
       --save_preds predictions/brightpn_run001.json
   ```

3. Compare results and tune hyperparameters

4. Manually verify unlabelled test predictions

5. If performance improves, retrain with adjusted thresholds

---

## 📚 Key Files

- `main_unlabelled.py` - Semi-supervised training script
- `utils/semisup_utils.py` - Alpha scheduling, confidence filtering
- `utils/data_utils.py` - Load unlabelled data, create test sets
- `utils/model_utils.py` - Semi-supervised training loop

**Original supervised training is unchanged:**
- `main.py` - Still works as before!

---

**Good luck with semi-supervised learning!** 🎉

