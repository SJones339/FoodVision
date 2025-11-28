# 🚀 Quick Start Guide: Training Your Models

## Where You Are Right Now

**Status**: Ready to train classification models! ✅

**Progress**:
- ✅ Detection model (YOLOv8) - Baseline trained
- ✅ Custom CNN architecture - Designed
- ✅ Training scripts - Created
- ⏳ **CURRENT**: Train Custom CNN & EfficientNet

---

## Option 1: Quick Training Script (Easiest)

**Run this command:**
```bash
python quick_train.py --epochs 20 --batch-size 32
```

This will:
1. Train Custom CNN (from scratch)
2. Train EfficientNet (transfer learning)
3. Compare results automatically
4. Save models to `models/` directory

**Time**: ~3-4 hours (mostly waiting)

**Options**:
```bash
# Train only Custom CNN
python quick_train.py --custom-only --epochs 20

# Train only EfficientNet
python quick_train.py --efficientnet-only --epochs 20

# Use CPU instead of GPU
python quick_train.py --device cpu

# Use fewer samples for faster training (testing)
python quick_train.py --max-samples 1000 --epochs 10
```

---

## Option 2: Use Jupyter Notebook

**Open the notebook:**
```bash
jupyter notebook notebooks/04_quick_start_training.ipynb
```

Then run cells step by step. This gives you more control and you can see progress in real-time.

---

## What Happens During Training

### Custom CNN Training:
1. Loads Food-101 dataset
2. Creates custom CNN model (no pretrained weights)
3. Trains for 20 epochs
4. Saves best model to `models/custom_cnn_best.pth`
5. Shows training progress and final accuracy

### EfficientNet Training:
1. Loads Food-101 dataset
2. Creates EfficientNet with ImageNet pretrained weights
3. Trains for 20 epochs (faster convergence expected)
4. Saves best model to `models/efficientnet_best.pth`
5. Shows training progress and final accuracy

### Comparison:
- Prints side-by-side comparison
- Shows accuracy difference
- Saves results to `models/training_results.json`

---

## Expected Results

| Model | Accuracy | Training Time | Notes |
|-------|----------|---------------|-------|
| Custom CNN | 60-75% | ~2-3 hours | Trained from scratch |
| EfficientNet | 80-90% | ~1-2 hours | Transfer learning |

**Why EfficientNet is better**: Pretrained weights help it learn faster and achieve higher accuracy.

---

## After Training: Next Steps

1. **Check Results**:
   ```bash
   ls -lh models/
   cat models/training_results.json
   ```

2. **Use Best Model**:
   - EfficientNet will likely be better
   - Use it for the classification pipeline

3. **Integrate with Detection**:
   - YOLOv8 detects food regions
   - EfficientNet classifies each region
   - Combine into end-to-end pipeline

---

## Troubleshooting

**Out of Memory (OOM)**:
```bash
# Reduce batch size
python quick_train.py --batch-size 16

# Use fewer samples
python quick_train.py --max-samples 2000
```

**Training Too Slow**:
```bash
# Use fewer epochs for testing
python quick_train.py --epochs 10

# Use fewer samples
python quick_train.py --max-samples 1000
```

**No GPU Available**:
```bash
# Will automatically use CPU, but add flag to be explicit
python quick_train.py --device cpu
```

---

## Files Created After Training

```
models/
├── custom_cnn_best.pth          # Custom CNN model
├── efficientnet_best.pth         # EfficientNet model
└── training_results.json         # Comparison results
```

---

## Where This Fits in Overall Plan

**Current Phase**: Day 1 - Classification Training

**Completed**:
- ✅ Detection baseline (YOLOv8)
- ✅ Custom CNN architecture
- ✅ Training infrastructure

**In Progress**:
- ⏳ Training Custom CNN
- ⏳ Training EfficientNet

**Next** (Day 2):
- Integrate detection + classification
- Create end-to-end pipeline
- Add nutrition API

**Final** (Day 3):
- Evaluation & metrics
- Presentation prep
- Demo

---

## Quick Commands Reference

```bash
# Start training both models
python quick_train.py

# Train with custom settings
python quick_train.py --epochs 30 --batch-size 16

# Check if models exist
ls models/

# View results
cat models/training_results.json
```

---

**You're all set!** Just run `python quick_train.py` and let it train. Check back in a few hours for results! 🎯

