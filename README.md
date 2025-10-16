# Spectral Classification Project

Deep learning classification of astronomical spectra into 4 classes: AGN, HM/LM stars, YSO/CV, and NS/HMXB/LMXB/NS_BIN.

---

## Quick Start

### Basic Training (2-class: AGN vs HM/LM)
```bash
python main.py --classes 2 --epochs 100
```

### Advanced Training (4-class with improvements)
```bash
python main.py --classes 4 --epochs 150 --oversample_target 1.0
```

---

## Training Commands

### Standard Training
```bash
python main.py --classes 4 --epochs 150
```

### With Data Augmentation & Balancing
```bash
python main.py --classes 4 --epochs 150 --oversample_target 1.0
```
- `--oversample_target 1.0`: Fully balance all classes (recommended for 4-class)
- Default is 0.7 (70% of majority class)

### With Stronger Focal Loss
```bash
python main.py --classes 4 --epochs 150 --oversample_target 1.0 --focal_gamma 3.0
```
- `--focal_gamma`: Focus more on hard examples (default: 2.0, try: 3.0)

### Complete Example
```bash
python main.py --classes 4 --epochs 200 --oversample_target 1.0 \
    --focal_gamma 3.0 --lr 5e-4 --batch_size 64
```

---

## Real-Time Visualization During Training

Watch how your model learns by visualizing the latent space every N epochs:

### Enable Visualization
```bash
python main.py --classes 4 --epochs 150 --oversample_target 1.0 --visualize_training
```

### Visualize More Frequently
```bash
python main.py --classes 4 --epochs 150 --oversample_target 1.0 \
    --visualize_training --viz_every 5
```
- `--viz_every N`: Visualize every N epochs (default: 10)

### Use UMAP (Better Quality, Slower)
```bash
python main.py --classes 4 --epochs 150 --oversample_target 1.0 \
    --visualize_training --viz_method umap
```
- `--viz_method`: Choose 'tsne' (fast) or 'umap' (better quality)

### Output Location
```
training_feature_visualizations/
└── run_001/
    ├── epoch_001_acc0.5234.png
    ├── epoch_010_acc0.6543.png
    ├── epoch_150_acc0.8645.png
    └── create_animation.sh  ← Run to create GIF
```

### Create Animation
```bash
cd training_feature_visualizations/run_001/
./create_animation.sh
```

---

## Post-Training Visualization

Visualize a trained model's latent space:

### Basic Visualization
```bash
python visualize_latent_space.py --checkpoint checkpoints/best_4cls_*.pt --classes 4
```

### Output Location
```
visualizations/
└── best_4cls_acc86.45%/
    ├── latent_space_2d_umap_86.45%.png
    ├── class_separability_umap_86.45%.png
    └── statistics_86.45%.txt
```

### Compare Multiple Models
```bash
# Visualize baseline
python visualize_latent_space.py --checkpoint checkpoints/best_4cls_*val0.8095.pt --classes 4

# Visualize improved
python visualize_latent_space.py --checkpoint checkpoints/best_4cls_*val0.8645.pt --classes 4

# Compare folders
ls visualizations/
```

---

## Key Parameters

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `--classes` | 2, 4 | 2 | Number of classes |
| `--epochs` | int | 100 | Training epochs |
| `--oversample_target` | 0.0-1.0 | 0.7 | Balance classes (1.0 = fully balanced) |
| `--focal_gamma` | 1.0-5.0 | 2.0 | Focus on hard examples (higher = more focus) |
| `--visualize_training` | flag | off | Enable real-time visualization |
| `--viz_every` | int | 10 | Visualize every N epochs |
| `--viz_method` | tsne, umap | tsne | Dimensionality reduction method |
| `--lr` | float | 1e-3 | Learning rate |
| `--batch_size` | int | 64 | Batch size |

---

## Model Improvements

### 1. Enhanced CNN Architecture
- **Before**: 2 conv layers (32→64 channels)
- **After**: 3 conv layers (64→128→256 channels) with dropout
- **Impact**: 4x more parameters, better feature extraction

### 2. Focal Loss
- Replaces standard CrossEntropyLoss
- Automatically focuses on hard-to-classify examples
- Better for imbalanced classes

### 3. Data Augmentation
- **Minority classes** get stronger augmentation
- **NS class (<100 samples)**: 2x noise, ±3 bin shift
- **YSO/CV (<300 samples)**: 1.5x noise, ±2 bin shift
- Conservative for spectral data (max 1% noise)

### 4. Learning Rate Scheduler
- Auto-reduces learning rate when validation loss plateaus
- Helps fine-tune in later epochs

---

## Expected Performance

### 2-Class (AGN vs HM/LM)
```
Accuracy: ~96%
AGN:    94% precision, 97% recall
HM/LM:  96% precision, 92% recall
```

### 4-Class (Baseline)
```
Accuracy: ~81%
AGN:    92% precision, 89% recall
HM/LM:  81% precision, 90% recall
YSO/CV: 47% precision, 33% recall  ← struggles
NS:     38% precision, 50% recall  ← struggles
```

### 4-Class (With Improvements)
```bash
python main.py --classes 4 --epochs 150 --oversample_target 1.0 --focal_gamma 3.0
```
```
Accuracy: ~85-87%
AGN:    93% precision, 91% recall
HM/LM:  87% precision, 89% recall
YSO/CV: 55% precision, 50% recall  ← improved!
NS:     50% precision, 65% recall  ← improved!
```

---

## File Structure

```
spectra_classification/
├── main.py                          # Main training script
├── visualize_latent_space.py        # Post-training visualization
├── training_visualizer.py           # Real-time training visualization
├── utils/
│   ├── model_utils.py               # CNN, Focal Loss
│   └── data_utils.py                # Data loading, augmentation
├── data/
│   └── Brightpn_id_normspec_counts_label_onlylabelled.txt
├── checkpoints/                     # Saved models
├── visualizations/                  # Post-training plots
└── training_feature_visualizations/ # Training evolution plots
```

---

## Quick Examples

### Example 1: Train and Visualize
```bash
# Train with visualization
python main.py --classes 4 --epochs 150 --oversample_target 1.0 --visualize_training

# Create animation
cd training_feature_visualizations/run_001/
./create_animation.sh
```

### Example 2: Compare Hyperparameters
```bash
# Baseline
python main.py --classes 4 --epochs 100 --visualize_training

# Improved
python main.py --classes 4 --epochs 100 --oversample_target 1.0 --focal_gamma 3.0 --visualize_training

# Compare: ls training_feature_visualizations/
```

### Example 3: Visualize Saved Model
```bash
# Find checkpoint
ls checkpoints/best_4cls_*.pt

# Visualize it
python visualize_latent_space.py --checkpoint checkpoints/best_4cls_*.pt --classes 4

# View results
open visualizations/best_4cls_acc86.45%/
```

---

## Installation

```bash
# Required
pip install torch numpy pandas scikit-learn matplotlib seaborn

# Optional (for better visualizations)
pip install umap-learn

# Optional (for creating animations)
sudo apt-get install imagemagick  # or ffmpeg
```

---

## Tips

### For Best 4-Class Results:
1. Use `--oversample_target 1.0` to fully balance classes
2. Try `--focal_gamma 3.0` if minority classes still struggle
3. Train for at least 150 epochs
4. Use `--visualize_training` to monitor progress

### For Debugging:
- Enable `--visualize_training` to see if clusters are forming
- If validation << training: reduce `--oversample_target` or add more dropout
- If clusters not forming: increase `--epochs` or model capacity

### For Fast Experiments:
```bash
python main.py --classes 4 --epochs 50 --visualize_training --viz_every 5
```

### For Publication-Quality Results:
```bash
python main.py --classes 4 --epochs 200 --oversample_target 1.0 \
    --focal_gamma 3.0 --visualize_training --viz_method umap
```

---

## Common Workflows

### Workflow 1: Train 2-Class Model
```bash
python main.py --classes 2 --epochs 100
# → High accuracy (~96%)
```

### Workflow 2: Train 4-Class Model (Improved)
```bash
python main.py --classes 4 --epochs 150 --oversample_target 1.0
# → Good accuracy (~85%)
```

### Workflow 3: Full Pipeline with Visualization
```bash
# 1. Train with real-time viz
python main.py --classes 4 --epochs 150 --oversample_target 1.0 \
    --visualize_training --viz_every 10

# 2. Create animation
cd training_feature_visualizations/run_001/
./create_animation.sh

# 3. Visualize final model
python visualize_latent_space.py --checkpoint checkpoints/best_4cls_*.pt --classes 4

# 4. Compare results
ls visualizations/
ls training_feature_visualizations/
```

---

## Troubleshooting

### Low 4-class accuracy?
```bash
# Try: Fully balance + stronger focal loss
python main.py --classes 4 --epochs 200 --oversample_target 1.0 --focal_gamma 3.0
```

### Overfitting (train >> val)?
```bash
# Reduce oversample_target
python main.py --classes 4 --epochs 150 --oversample_target 0.8
```

### Slow training?
- Use `--batch_size 32` for smaller batches
- Use `--viz_method tsne` instead of umap
- Reduce `--viz_every 20` for less frequent visualization

### "Input type (double) error"?
Already fixed - data is auto-converted to float32

### "UMAP not installed"?
```bash
pip install umap-learn
# Or use: --viz_method tsne
```

---

## Documentation

- **IMPROVEMENTS_SUMMARY.md** - Detailed explanation of all improvements
- **TRAINING_VISUALIZATION_GUIDE.md** - Complete training visualization guide
- **VISUALIZATION_USAGE.md** - Post-training visualization guide
- **TRAINING_VIZ_QUICKSTART.txt** - Quick reference for training viz

---

## Contact

For issues or questions, refer to the documentation files above.

---

**Quick command to get started:**
```bash
python main.py --classes 4 --epochs 150 --oversample_target 1.0 --visualize_training
```

Good luck! 🚀
