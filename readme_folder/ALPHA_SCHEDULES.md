# Alpha Scheduling Options

Choose the right alpha schedule based on how quickly your model learns.

---

## 📊 Schedule Comparison

### **Standard Schedule (Default)**
**Use when:** Model learns gradually, needs warm-up
```bash
python main_unlabelled.py --alpha_schedule standard --alpha_warmup 50
```

| Epoch | Alpha | Description |
|-------|-------|-------------|
| 1-50  | 0.0   | Supervised only (warm-up) |
| 51    | 0.01  | Start adding pseudo-labels |
| 100   | 0.27  | 1/3 of max alpha |
| 150   | 0.53  | 2/3 of max alpha |
| 200   | 0.8   | Full alpha (max) |

**Timeline:**
```
Epoch:   1 -------- 50 --------- 100 -------- 150 -------- 200
Alpha:   0.0        0.0          0.27         0.53         0.8
         └──────────┘└────────────────────────────────────┘
         Supervised    Gradual pseudo-label incorporation
```

---

### **Immediate Schedule** ⚡
**Use when:** Model learns FAST (reaches 90%+ by epoch 30)
```bash
python main_unlabelled.py --alpha_schedule immediate --alpha_warmup 10
```

| Epoch | Alpha | Description |
|-------|-------|-------------|
| 1     | 0.1   | Start with pseudo-labels immediately! |
| 5     | 0.2   | Gentle ramp-up |
| 10    | 0.3   | End of mini warm-up |
| 50    | 0.5   | Halfway to max |
| 100   | 0.65  | Most of max alpha |
| 200   | 0.8   | Full alpha (max) |

**Timeline:**
```
Epoch:   1 -- 10 ---------- 50 ---------- 100 ---------- 200
Alpha:   0.1  0.3           0.5           0.65           0.8
         └───┘└──────────────────────────────────────────┘
         Quick   Continuous pseudo-label incorporation
         warmup
```

---

## 🎯 When to Use Each

### Use **Standard** if:
- ✅ Model accuracy < 80% by epoch 30
- ✅ First time training on this dataset
- ✅ 4-class mode (harder problem)
- ✅ Conservative approach (safer)

### Use **Immediate** if:
- ⚡ Model accuracy > 90% by epoch 30
- ⚡ 2-class mode (easier problem)
- ⚡ Model learns very fast
- ⚡ You want to leverage unlabelled data sooner

---

## 📈 Your Case Study

**Your results:**
```
Epoch 027 | alpha=0.000 | val_acc=0.9406  ← 94% accuracy!
Epoch 030 | alpha=0.000 | val_acc=0.9267
```

**Analysis:**
- Model reached 94% by epoch 27
- Alpha was still 0.0 (no pseudo-labels yet)
- Wasting 23 epochs before pseudo-labels kick in!

**Solution:** Use **immediate** schedule! ⚡

```bash
python main_unlabelled.py --classes 2 --epochs 150 --data Brightpn \
    --alpha_schedule immediate --alpha_warmup 10
```

**Expected behavior:**
```
Epoch 001 | alpha=0.100 | val_acc=0.70  ← Start with pseudo-labels
Epoch 010 | alpha=0.300 | val_acc=0.88
Epoch 030 | alpha=0.42  | val_acc=0.94  ← Pseudo-labels already helping!
Epoch 100 | alpha=0.65  | val_acc=0.96
Epoch 150 | alpha=0.8   | val_acc=0.97  ← Benefit from unlabelled data
```

---

## 🔧 Customization

### Adjust Warmup Period
```bash
# Very fast learner: minimal warmup
--alpha_schedule immediate --alpha_warmup 5

# Still needs some warmup: moderate
--alpha_schedule immediate --alpha_warmup 20

# Conservative with immediate start
--alpha_schedule immediate --alpha_warmup 30
```

### Adjust Max Alpha
```bash
# Conservative: less weight on pseudo-labels
--alpha_max 0.6

# Balanced (default)
--alpha_max 0.8

# Note: Don't go higher than 0.8!
```

---

## 📊 Visual Comparison

### Standard Schedule (warmup=50, max_alpha=0.8)
```
1.0 |
0.8 |                                         ___________
0.6 |                            ___________/
0.4 |                ___________/
0.2 |   ___________/
0.0 |___/
    +--------------------------------------------------------
    0          50         100        150        200  (epoch)
```

### Immediate Schedule (warmup=10, max_alpha=0.8)
```
1.0 |
0.8 |                                         ___________
0.6 |                  ______________________/
0.4 |         ________/
0.2 |  ______/
0.0 |__/
    +--------------------------------------------------------
    0    10         50         100        150        200  (epoch)
```

---

## 🧪 Experimentation Tips

### Test Both Schedules

**Run 1: Standard (baseline)**
```bash
python main_unlabelled.py --classes 2 --epochs 150 \
    --alpha_schedule standard --alpha_warmup 50 \
    --save_preds predictions/standard_run.json
```

**Run 2: Immediate (experiment)**
```bash
python main_unlabelled.py --classes 2 --epochs 150 \
    --alpha_schedule immediate --alpha_warmup 10 \
    --save_preds predictions/immediate_run.json
```

**Compare:**
- Final validation accuracy
- Training time to convergence
- Test set performance

---

## 🚨 Warning Signs

### If using **immediate** and:
- ❌ Validation accuracy drops suddenly → Model unstable, use standard
- ❌ Training loss increases → Pseudo-labels hurting, reduce alpha_max
- ❌ Pseudo-accept rate very low (<10%) → Thresholds too high

### Solutions:
1. Switch back to standard schedule
2. Increase warmup: `--alpha_warmup 20`
3. Reduce max alpha: `--alpha_max 0.6`
4. Lower confidence thresholds

---

## 💡 Recommendations

### For 2-Class (AGN vs HM/LM/YSO):
```bash
# Model learns fast, use immediate
python main_unlabelled.py --classes 2 --epochs 150 \
    --alpha_schedule immediate --alpha_warmup 10 --alpha_max 0.8
```

### For 4-Class (harder problem):
```bash
# Start conservative, then experiment
python main_unlabelled.py --classes 4 --epochs 200 \
    --alpha_schedule standard --alpha_warmup 40 --alpha_max 0.8

# If model learns fast (>85% by epoch 30), try immediate
python main_unlabelled.py --classes 4 --epochs 200 \
    --alpha_schedule immediate --alpha_warmup 20 --alpha_max 0.7
```

---

## 📋 Quick Reference

| Schedule | Warmup | Start Alpha | Best For |
|----------|--------|-------------|----------|
| **standard** | 50 | 0.0 | Conservative, 4-class, first time |
| **immediate** | 10 | 0.1 | Fast learners, 2-class, experienced |

---

**Your Next Command:**

Since your model reached 94% by epoch 27, use **immediate**! ⚡

```bash
python main_unlabelled.py --classes 2 --epochs 150 --data Brightpn \
    --alpha_schedule immediate --alpha_warmup 10 \
    --save_preds predictions/immediate_brightpn_2cls.json
```

Good luck! 🚀

