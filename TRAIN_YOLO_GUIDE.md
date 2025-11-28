# Training YOLOv8 on Food Data

## Why Train YOLOv8?

**Current Problem**: Using COCO pretrained YOLOv8 (trained on general objects like person, car, dog) → Poor food detection

**Solution**: Train YOLOv8 on FoodSeg103 (food-specific dataset) → Better food detection

---

## What Training Does

### Before Training (COCO Pretrained):
- ❌ Doesn't know what food looks like
- ❌ Detects generic "objects" not food regions
- ❌ Poor bounding boxes around food
- ❌ Bad crops → Bad classification results

### After Training (FoodSeg103):
- ✅ Understands food appearance
- ✅ Better at finding food regions
- ✅ More accurate bounding boxes
- ✅ Better crops → Better classification results

---

## Important: YOLOv8 Classification vs EfficientNet

### YOLOv8 Can Classify, But...
- **Accuracy**: ~40-50% on fine-grained classification
- **Purpose**: Detection is primary, classification is secondary
- **Classes**: 103 FoodSeg103 classes (different from Food-101)

### EfficientNet Classification:
- **Accuracy**: 85% on Food-101
- **Purpose**: Optimized specifically for classification
- **Classes**: 101 Food-101 classes

### Why We Still Need EfficientNet:
**2x better accuracy** (85% vs ~40-50%) justifies the two-stage approach.

**YOLOv8**: Finds WHERE food is (detection)  
**EfficientNet**: Identifies WHAT food is (classification)

See `ARCHITECTURE_EXPLANATION.md` for full details.

---

## How to Train YOLOv8

### Prerequisites:
1. FoodSeg103 dataset converted to YOLO format
2. Data YAML file at `data/foodseg_pp/data.yaml`

### Step 1: Convert FoodSeg103 (if not done)
```bash
python download_datasets.py --foodseg103 --convert-yolo
```

### Step 2: Train YOLOv8
```bash
# Basic training (50 epochs, YOLOv8s)
python src/detection/train_yolo.py --epochs 50 --model-size s

# Faster training (fewer epochs, smaller model)
python src/detection/train_yolo.py --epochs 30 --model-size n

# Better accuracy (more epochs, larger model)
python src/detection/train_yolo.py --epochs 100 --model-size m
```

### Step 3: Use Trained Model
```python
from pipeline.end_to_end import FoodVisionPipeline

pipeline = FoodVisionPipeline(
    detection_model_path='models/yolov8s_food_best.pt',  # Your trained model
    classification_model_path='models/efficientnet_best.pth'
)
```

---

## Training Options

| Model Size | Speed | Accuracy | Use Case |
|------------|-------|----------|----------|
| **n** (nano) | Fastest | Lowest | Quick testing |
| **s** (small) | Fast | Good | **Recommended** |
| **m** (medium) | Medium | Better | Better accuracy |
| **l** (large) | Slow | Best | Maximum accuracy |

**Recommendation**: Start with `--model-size s` (good balance)

---

## Expected Results

### Before Training (COCO):
- mAP50: ~20-30% (on food data)
- Poor bounding boxes
- Misses many food items

### After Training (FoodSeg103, 50 epochs):
- mAP50: ~40-50% (2x improvement)
- Better bounding boxes
- Finds more food items
- More accurate crops for classification

---

## Training Time

- **YOLOv8n**: ~2-3 hours (50 epochs)
- **YOLOv8s**: ~4-6 hours (50 epochs) ← Recommended
- **YOLOv8m**: ~8-12 hours (50 epochs)

**Note**: Training on Apple Silicon (MPS) is faster than CPU.

---

## Troubleshooting

**"Dataset YAML not found"**
- Run: `python download_datasets.py --foodseg103 --convert-yolo`

**"Out of memory"**
- Reduce batch size: `--batch-size 8`
- Use smaller model: `--model-size n`

**"Training too slow"**
- Use GPU/MPS: `--device mps` (auto-detected)
- Reduce epochs: `--epochs 30`

---

## After Training

The trained model will be saved to:
- `models/yolov8s_food_best.pt` (or yolov8n, yolov8m, etc.)

Update your pipeline to use it:
```python
pipeline = FoodVisionPipeline(
    detection_model_path='models/yolov8s_food_best.pt'
)
```

**Expected Improvement**: Better bounding boxes → Better crops → Better classification results!

