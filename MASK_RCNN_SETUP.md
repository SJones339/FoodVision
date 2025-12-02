# Mask R-CNN Setup Guide

## Why Mask R-CNN?

**Current YOLOv8 Performance**: 34% mAP  
**Expected Mask R-CNN Performance**: 50-70% mAP  

Mask R-CNN is a state-of-the-art instance segmentation model that typically achieves **significantly better accuracy** than YOLOv8 for food detection.

## Installation

```bash
# Install PyTorch (if not already installed)
pip install torch torchvision

# Verify installation
python -c "import torch; import torchvision; print('✓ Installed')"
```

## Training

### Quick Start

```bash
# Make sure you have the dataset
python download_datasets.py --foodseg103 --convert-yolo

# Train Mask R-CNN
python train_mask_rcnn.py
```

### Training Options

```bash
# Custom training
python train_mask_rcnn.py \
    --epochs 100 \
    --batch-size 4 \
    --lr 0.001 \
    --device mps
```

### Expected Training Time

- **Epochs**: 50-100 (recommended)
- **Time per epoch**: ~30-60 minutes (on MPS/GPU)
- **Total time**: ~1-2 days for 50 epochs

## Performance Comparison

| Model | mAP50 | Training Time | Inference Speed |
|-------|-------|---------------|-----------------|
| YOLOv8m | 34% | 2 days | Fast (~30 FPS) |
| Mask R-CNN | **50-70%** | 1-2 days | Medium (~10 FPS) |

## Usage

### Inference

```python
from src.detection.mask_rcnn_inference import MaskRCNNFoodDetector

# Load trained model
detector = MaskRCNNFoodDetector(
    model_path='models/mask_rcnn_food_best.pth',
    conf_threshold=0.5
)

# Detect foods
results = detector.detect('meal.jpg')

for det in results:
    print(f"Food: {det['class_id']}")
    print(f"Confidence: {det['confidence']:.2f}")
    print(f"BBox: {det['bbox']}")
```

### Integration with Pipeline

To use Mask R-CNN in your pipeline, you'll need to update `src/pipeline/end_to_end.py` to use `MaskRCNNFoodDetector` instead of YOLO.

## Advantages of Mask R-CNN

1. ✅ **Higher accuracy** (50-70% vs 34% mAP)
2. ✅ **Better segmentation masks**
3. ✅ **More accurate bounding boxes**
4. ✅ **State-of-the-art architecture**

## Disadvantages

1. ⚠️ **Slower inference** (~10 FPS vs ~30 FPS)
2. ⚠️ **Larger model size**
3. ⚠️ **More memory intensive**
4. ⚠️ **More complex training**

## Recommendation

**Use Mask R-CNN if:**
- ✅ You need 70%+ mAP (production quality)
- ✅ Accuracy is more important than speed
- ✅ You have time for longer training

**Stick with YOLOv8 if:**
- ✅ Speed is critical (real-time applications)
- ✅ 34% mAP is acceptable
- ✅ You need smaller model size

## Next Steps

1. Train Mask R-CNN: `python train_mask_rcnn.py`
2. Evaluate on validation set
3. Compare with YOLOv8 results
4. Integrate into pipeline if better


