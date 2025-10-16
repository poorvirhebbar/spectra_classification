# Latent Space Visualization - Updated Guide

## 🎯 What's New

The visualization script now **automatically organizes outputs** based on your checkpoint name!

### Features:
- ✅ **Auto-creates folders** based on checkpoint accuracy and metadata
- ✅ **Parses checkpoint name** to extract accuracy, number of classes, etc.
- ✅ **Saves all outputs** in organized subdirectories
- ✅ **Generates summary statistics** file automatically

---

## 📁 Directory Structure

When you run the visualization, it creates:

```
visualizations/
└── best_4cls_acc85.67%/
    ├── latent_space_2d_umap_85.67%.png
    ├── class_separability_umap_85.67%.png
    └── statistics_85.67%.txt
```

**Example for different checkpoints:**
```
visualizations/
├── best_2cls_acc96.06%/
│   ├── latent_space_2d_umap_96.06%.png
│   ├── class_separability_umap_96.06%.png
│   └── statistics_96.06%.txt
│
├── best_4cls_acc81.23%/
│   ├── latent_space_2d_umap_81.23%.png
│   ├── class_separability_umap_81.23%.png
│   └── statistics_81.23%.txt
│
└── best_4cls_acc86.45%/  ← After improvements!
    ├── latent_space_2d_umap_86.45%.png
    ├── class_separability_umap_86.45%.png
    └── statistics_86.45%.txt
```

---

## 🚀 Usage

### Basic Command:
```bash
python visualize_latent_space.py --checkpoint checkpoints/best_4cls_*.pt --classes 4
```

### What It Does:

1. **Reads checkpoint name**: `best_4cls_2730ex_380bins_val0.8567.pt`
2. **Extracts metadata**:
   - Type: `best`
   - Classes: `4cls`
   - Accuracy: `85.67%`
3. **Creates folder**: `visualizations/best_4cls_acc85.67%/`
4. **Saves outputs** with meaningful names

---

## 📊 Output Files

### 1. `latent_space_2d_umap_85.67%.png`
Shows your latent space in 2D:
- **Left panel**: All samples colored by true class
- **Right panel**: Misclassifications highlighted with red borders

### 2. `class_separability_umap_85.67%.png`
Matrix showing class overlaps:
- **Diagonal**: Density of each class
- **Off-diagonal**: Overlap between class pairs

### 3. `statistics_85.67%.txt`
Text summary with:
- Per-class accuracy
- Most confused class pairs
- Total samples analyzed
- Checkpoint metadata

---

## 💡 Examples

### Compare Before/After Improvements:

```bash
# Visualize baseline model (80.95% accuracy)
python visualize_latent_space.py \
    --checkpoint checkpoints/best_4cls_2730ex_380bins_val0.8095.pt \
    --classes 4

# Visualize improved model (86.45% accuracy)
python visualize_latent_space.py \
    --checkpoint checkpoints/best_4cls_4360ex_380bins_val0.8645.pt \
    --classes 4
```

**Result:**
```
visualizations/
├── best_4cls_acc80.95%/  ← Before
└── best_4cls_acc86.45%/  ← After
```

Now you can easily compare side-by-side!

---

### Visualize 2-class vs 4-class:

```bash
# 2-class model
python visualize_latent_space.py \
    --checkpoint checkpoints/best_2cls_*.pt \
    --classes 2

# 4-class model
python visualize_latent_space.py \
    --checkpoint checkpoints/best_4cls_*.pt \
    --classes 4
```

**Result:**
```
visualizations/
├── best_2cls_acc96.06%/
└── best_4cls_acc86.45%/
```

---

### Use t-SNE instead of UMAP:

```bash
python visualize_latent_space.py \
    --checkpoint checkpoints/best_4cls_*.pt \
    --classes 4 \
    --method tsne
```

**Result:**
```
visualizations/
└── best_4cls_acc86.45%/
    ├── latent_space_2d_tsne_86.45%.png  ← t-SNE instead of UMAP
    ├── class_separability_tsne_86.45%.png
    └── statistics_86.45%.txt
```

---

## 📈 Workflow Example

### Complete workflow from training to visualization:

```bash
# 1. Train baseline model
python main.py --classes 4 --epochs 100
# Output: checkpoints/best_4cls_2730ex_380bins_val0.8095.pt

# 2. Visualize baseline
python visualize_latent_space.py --checkpoint checkpoints/best_4cls_*val0.8095.pt --classes 4
# Output: visualizations/best_4cls_acc80.95%/

# 3. Train improved model
python main.py --classes 4 --epochs 150 --oversample_target 1.0
# Output: checkpoints/best_4cls_4360ex_380bins_val0.8645.pt

# 4. Visualize improved
python visualize_latent_space.py --checkpoint checkpoints/best_4cls_*val0.8645.pt --classes 4
# Output: visualizations/best_4cls_acc86.45%/

# 5. Compare folders side-by-side!
ls -lh visualizations/
```

---

## 🔍 Understanding the Output

### Example Console Output:

```
📁 Output directory: visualizations/best_4cls_acc86.45%
   Checkpoint info: {'type': 'best', 'classes': '4cls', 
                     'accuracy': '0.8645', 'accuracy_pct': '86.45%'}

Loading checkpoint from checkpoints/best_4cls_4360ex_380bins_val0.8645.pt...
Loading data from data/Brightpn_id_normspec_counts_label_onlylabelled.txt...
Using 2730 samples across 4 classes

Loading model...
Extracting features from validation set...
Extracted features shape: (546, 128)

============================================================
LATENT SPACE STATISTICS
============================================================
Total samples: 546
Overall accuracy: 0.8645

Per-class performance:
  AGN                 : 249/268 correct (92.91%)
  HM/LM               : 172/189 correct (91.01%)
  YSO/CV              :  38/ 69 correct (55.07%)
  NS/HMXB/LMXB/NS_BIN :  13/ 20 correct (65.00%)

Most confused pairs:
  YSO/CV          → HM/LM         :  18 errors
  YSO/CV          → AGN           :  10 errors
  NS/HMXB/LMXB/NS_BIN → AGN       :   5 errors
============================================================

Reducing to 2D using UMAP...
Creating visualizations...

✅ Visualization complete!

📂 Saved to: visualizations/best_4cls_acc86.45%/
   📊 latent_space_2d_umap_86.45%.png
   📊 class_separability_umap_86.45%.png
   📄 statistics_86.45%.txt

To view: open visualizations/best_4cls_acc86.45%/
```

---

## 📝 Summary Statistics File

The `statistics_*.txt` file contains:

```
============================================================
LATENT SPACE VISUALIZATION SUMMARY
============================================================
Checkpoint: best_4cls_4360ex_380bins_val0.8645.pt
Output directory: visualizations/best_4cls_acc86.45%
Method: UMAP
Classes: 4
Validation accuracy: 86.45%
Total samples: 546

Per-class performance:
  AGN                 : 249/268 correct (92.91%)
  HM/LM               : 172/189 correct (91.01%)
  YSO/CV              :  38/ 69 correct (55.07%)
  NS/HMXB/LMXB/NS_BIN :  13/ 20 correct (65.00%)

Most confused pairs:
  YSO/CV          → HM/LM         :  18 errors
  YSO/CV          → AGN           :  10 errors
  NS/HMXB/LMXB/NS_BIN → AGN       :   5 errors
============================================================
```

Perfect for including in papers or presentations!

---

## 🎨 Customization

### Change Output Directory:

Edit line 97 in `visualize_latent_space.py`:
```python
vis_dir = Path("visualizations")  # Change to "my_plots"
```

### Change Naming Convention:

Edit lines 102-103:
```python
subdir_name = f"{info['type']}_{info.get('classes', 'unknown')}_acc{info['accuracy_pct']}"
```

---

## 🐛 Troubleshooting

### Issue: "No such file or directory: 'visualizations/...'"
**Solution:** The script creates it automatically. No action needed.

### Issue: Checkpoint name doesn't parse correctly
**Example:** Custom checkpoint name like `my_model.pt`
**Solution:** The script falls back to using the full filename as the folder name.

### Issue: Want to compare multiple runs
**Solution:** Just run the script multiple times with different checkpoints. Each gets its own folder!

---

## 🎯 Quick Reference

```bash
# Basic usage
python visualize_latent_space.py --checkpoint checkpoints/best_*.pt --classes 4

# With t-SNE
python visualize_latent_space.py --checkpoint checkpoints/best_*.pt --classes 4 --method tsne

# 2-class model
python visualize_latent_space.py --checkpoint checkpoints/best_2cls_*.pt --classes 2

# View all visualizations
ls -lh visualizations/

# Open specific visualization
open visualizations/best_4cls_acc86.45%/latent_space_2d_umap_86.45%.png
```

---

Happy visualizing! 🎨

