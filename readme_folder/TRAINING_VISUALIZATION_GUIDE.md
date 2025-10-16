# Training Visualization Guide 📊

Watch your model learn in real-time! This feature creates visualizations of the latent space at regular intervals during training, showing how feature representations evolve over time.

---

## 🎯 What You'll See

### During Training:
- **Epoch-by-epoch snapshots** of how the model organizes classes in latent space
- **Training set**: How well features are being learned
- **Validation set**: Which samples are being misclassified
- **Evolution over time**: Watch clusters form and tighten

### Example Output:
```
training_feature_visualizations/
└── run_001/
    ├── epoch_001_acc0.5234.png  ← Early: scattered
    ├── epoch_010_acc0.6543.png
    ├── epoch_025_acc0.7456.png
    ├── epoch_050_acc0.8123.png
    ├── epoch_100_acc0.8567.png  ← Later: clustered!
    ├── epoch_150_acc0.8645.png
    ├── run_metadata.json
    ├── training_progress.txt
    └── create_animation.sh
```

---

## 🚀 Quick Start

### Basic Usage:
```bash
python main.py --classes 4 --epochs 150 --oversample_target 1.0 --visualize_training
```

### With Custom Frequency:
```bash
# Visualize every 5 epochs
python main.py --classes 4 --epochs 150 --oversample_target 1.0 \
    --visualize_training --viz_every 5
```

### With UMAP (better but slower):
```bash
python main.py --classes 4 --epochs 150 --oversample_target 1.0 \
    --visualize_training --viz_method umap
```

---

## 📊 Parameters

### `--visualize_training`
Enable real-time visualization (default: disabled)

### `--viz_every N`
Visualize every N epochs (default: 10)

**Recommendations:**
- **Short training (<50 epochs):** `--viz_every 5`
- **Normal training (100-150 epochs):** `--viz_every 10`
- **Long training (200+ epochs):** `--viz_every 25`

### `--viz_method {umap, tsne}`
Dimensionality reduction method (default: tsne)

**Comparison:**
- **t-SNE**: Faster, good for quick iterations (recommended)
- **UMAP**: Better structure preservation, slower

---

## 📁 Directory Structure

### Automatic Organization:
Each training run gets its own folder:

```
training_feature_visualizations/
├── run_001/  ← First training run
│   ├── epoch_001_acc0.5234.png
│   ├── epoch_010_acc0.6543.png
│   ├── ...
│   ├── epoch_150_acc0.8645.png
│   ├── run_metadata.json
│   ├── training_progress.txt
│   └── create_animation.sh
│
├── run_002/  ← Second training run (different hyperparameters)
│   ├── epoch_001_acc0.5123.png
│   ├── ...
│
└── run_003/  ← Third training run
    ├── epoch_001_acc0.5456.png
    ├── ...
```

**Key Feature:** Previous runs are NEVER deleted! 🎉

---

## 🖼️ Understanding the Visualizations

### Left Panel: Training Set
Shows how the model is learning features:
- **Points colored by true class**
- Early epochs: Scattered, no clear clusters
- Later epochs: Tight clusters forming

### Right Panel: Validation Set
Shows generalization:
- **✓ Correct predictions**: Normal markers
- **✗ Misclassifications**: Large red X markers
- Watch misclassifications decrease over time!

### What to Look For:

#### Early Training (Epochs 1-25):
```
[All classes mixed together]
  ●○△▲●○△▲
  ○△▲●○△▲●
  △▲●○△▲●○

Random initialization - no structure yet
```

#### Mid Training (Epochs 25-75):
```
[Clusters starting to form]
  ●●●●    ○○○○
  ●●●●    ○○○○
          △△  ▲▲
        △△△△  ▲▲

Majority classes separating first
```

#### Late Training (Epochs 75+):
```
[Clear separation]
  ●●●●        ○○○○
  ●●●●        ○○○○
  
     △△△    ▲▲▲
     △△△    ▲▲▲

All classes well-separated!
```

---

## 📈 Interpreting Results

### Good Training Progress:

✅ **Clusters tightening over time**
```
Epoch 10: Spread = 5.0 units
Epoch 50: Spread = 2.5 units
Epoch 100: Spread = 1.0 units
```

✅ **Classes moving apart**
```
Epoch 10: AGN ↔ HM/LM distance = 1.0
Epoch 100: AGN ↔ HM/LM distance = 3.5
```

✅ **Misclassifications decreasing**
```
Epoch 10: 150 errors (Red X's everywhere)
Epoch 100: 30 errors (Few Red X's)
```

### Warning Signs:

⚠️ **Clusters not forming**
- Classes remain scattered after 50+ epochs
- **Solution**: Increase model capacity, check data quality

⚠️ **Minority classes embedded in majority**
- YSO/CV points inside AGN cluster
- **Solution**: Increase `--oversample_target`, stronger augmentation

⚠️ **Validation worse than training**
- Training clusters tight, validation scattered
- **Solution**: Overfitting - reduce model size, add regularization

---

## 🎬 Creating Animations

After training completes, create an animated GIF or video:

### Option 1: Using ImageMagick (GIF)
```bash
cd training_feature_visualizations/run_001/
./create_animation.sh
```

**Output:** `training_animation.gif`

### Option 2: Using FFmpeg (MP4)
```bash
cd training_feature_visualizations/run_001/
ffmpeg -framerate 2 -pattern_type glob -i 'epoch_*.png' \
       -c:v libx264 -pix_fmt yuv420p training_evolution.mp4
```

**Output:** `training_evolution.mp4`

### Installation:
```bash
# ImageMagick
sudo apt-get install imagemagick

# FFmpeg
sudo apt-get install ffmpeg
```

---

## 📊 Example Workflows

### Workflow 1: Quick Experiment
```bash
# Train for 50 epochs, visualize every 5
python main.py --classes 4 --epochs 50 \
    --visualize_training --viz_every 5
```

**Result:** 10-12 visualizations showing early learning

---

### Workflow 2: Full Training with Monitoring
```bash
# Train for 150 epochs, visualize every 10
python main.py --classes 4 --epochs 150 --oversample_target 1.0 \
    --visualize_training --viz_every 10
```

**Result:** ~15 visualizations showing complete evolution

---

### Workflow 3: Compare Hyperparameters
```bash
# Run 1: Baseline
python main.py --classes 4 --epochs 100 --oversample_target 0.7 \
    --visualize_training
# → Saves to run_001/

# Run 2: Improved
python main.py --classes 4 --epochs 100 --oversample_target 1.0 \
    --visualize_training
# → Saves to run_002/

# Compare side-by-side!
```

---

### Workflow 4: Long Training with UMAP
```bash
# Train for 200 epochs with high-quality viz
python main.py --classes 4 --epochs 200 --oversample_target 1.0 \
    --focal_gamma 3.0 --visualize_training --viz_every 25 --viz_method umap
```

**Result:** ~8 high-quality UMAP visualizations

---

## 📝 Files Explained

### `epoch_XXX_accY.YYYY.png`
Visualization for epoch XXX with validation accuracy Y.YYYY

### `run_metadata.json`
Metadata about the training run:
```json
{
  "run_directory": "training_feature_visualizations/run_001",
  "num_classes": 4,
  "method": "tsne",
  "start_time": "2025-10-16T14:30:00",
  "has_umap": false
}
```

### `training_progress.txt`
Tab-delimited training log:
```
Epoch | Train Acc | Val Acc | Train Loss | Val Loss
------------------------------------------------------------
    1 |    0.5234 |  0.4987 |     1.2345 |   1.3456
   10 |    0.6543 |  0.6234 |     0.9876 |   1.0234
   25 |    0.7456 |  0.7123 |     0.6543 |   0.7890
  ...
```

### `create_animation.sh`
Shell script to generate animations (auto-generated)

---

## 💡 Tips & Tricks

### Tip 1: Balance Speed vs Quality
- **Development**: Use `--viz_every 10` with `--viz_method tsne`
- **Final runs**: Use `--viz_every 5` with `--viz_method umap`

### Tip 2: Visualize at Key Milestones
The script automatically visualizes at:
- Epoch 1 (initialization)
- Epochs 5, 10, 25, 50, 75, 100 (milestones)
- Every N epochs (`--viz_every`)
- Last epoch (final result)

### Tip 3: Compare Before/After
```bash
# Before improvements
python main.py --classes 4 --epochs 100 --visualize_training
# Check: training_feature_visualizations/run_001/

# After improvements
python main.py --classes 4 --epochs 100 --oversample_target 1.0 \
    --focal_gamma 3.0 --visualize_training
# Check: training_feature_visualizations/run_002/

# Compare epoch_100 from both runs!
```

### Tip 4: Monitor Training Remotely
Upload images to cloud storage or sync to local machine:
```bash
# On server, after each epoch
rsync -av training_feature_visualizations/ user@local:/path/to/local/
```

---

## ⚡ Performance Impact

### Speed Comparison:
| Configuration | Time per Epoch | Overhead |
|--------------|----------------|----------|
| No visualization | 30s | 0% |
| With t-SNE (every 10) | 32s | ~7% |
| With UMAP (every 10) | 35s | ~17% |

**Recommendation:** The overhead is minimal! Enable it for all important runs.

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'sklearn'"
```bash
pip install scikit-learn
```

### Issue: "UMAP not found, using t-SNE"
```bash
pip install umap-learn
```

### Issue: Too many/few visualizations
Adjust `--viz_every`:
- More frequent: `--viz_every 5`
- Less frequent: `--viz_every 25`

### Issue: Visualizations look crowded
The script automatically limits to 500 training + 300 validation samples for speed. This is fine for seeing overall structure.

### Issue: Animation script fails
Check if imagemagick is installed:
```bash
which convert
# If not found:
sudo apt-get install imagemagick
```

---

## 📚 Example Output Analysis

### Scenario 1: Successful Training
```
Epoch 1:   Scattered, val_acc = 52%
Epoch 25:  Some clusters, val_acc = 71%
Epoch 50:  Clear clusters, val_acc = 82%
Epoch 100: Tight clusters, val_acc = 86%
```
**Interpretation:** ✅ Model learning well, continue training

### Scenario 2: Overfitting
```
Epoch 1:   Scattered, train_acc = 52%, val_acc = 50%
Epoch 50:  Tight train clusters, scattered val, train_acc = 90%, val_acc = 75%
Epoch 100: Very tight train, scattered val, train_acc = 95%, val_acc = 74%
```
**Interpretation:** ⚠️ Overfitting after epoch 50, stop early or add regularization

### Scenario 3: Underfitting
```
Epoch 1:   Scattered, val_acc = 52%
Epoch 50:  Still scattered, val_acc = 58%
Epoch 100: Still scattered, val_acc = 62%
```
**Interpretation:** ⚠️ Model capacity too small, increase model size

---

## 🎓 Advanced: Custom Visualization

Want to customize? Edit `training_visualizer.py`:

```python
# Change colors
CLASS_COLORS = {0: '#FF0000', 1: '#00FF00', ...}

# Change visualization frequency
def should_visualize(epoch, total_epochs, visualize_every=10):
    # Your custom logic
    ...

# Change plot style
ax.scatter(..., s=100, alpha=0.8)  # Larger, more opaque
```

---

## 📽️ Share Your Results

Create a compelling visualization for presentations:

1. Train with visualization enabled
2. Create animation:
   ```bash
   cd training_feature_visualizations/run_001/
   ./create_animation.sh
   ```
3. Share `training_animation.gif` in presentations!

---

Happy training! Watch your model learn! 🚀📊

