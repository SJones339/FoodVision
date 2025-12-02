# Mask R-CNN Training: File Structure

## Main Files Overview

### 1. **Entry Point: `train_mask_rcnn.py`**
**Location**: Root directory  
**Purpose**: Simple script to start training  
**What it does**:
- Checks if dataset exists
- Calls the main training function
- Sets training parameters (epochs, batch size, etc.)

**Key parameters**:
- `epochs=30` - Number of training epochs
- `batch_size=2` - Batch size (reduced for speed)
- `device='cpu'` - Device to use (CPU often faster than MPS for Mask R-CNN)
- `data_yaml='datasets/foodseg_pp/data.yaml'` - Dataset configuration

---

### 2. **Core Training Code: `src/detection/train_mask_rcnn.py`**
**Location**: `src/detection/train_mask_rcnn.py`  
**Purpose**: Main training implementation  
**Key Components**:

#### A. **FoodSegDataset Class** (Lines ~30-130)
- **Purpose**: Loads and preprocesses training data
- **Input**: YOLO format dataset (images + labels)
- **Output**: PyTorch tensors (images, bounding boxes, masks, labels)
- **Key methods**:
  - `__getitem__()`: Loads one image and its annotations
  - Converts YOLO format → Mask R-CNN format
  - Validates bounding boxes (filters invalid ones)

#### B. **get_model_instance_segmentation()** (Lines ~130-150)
- **Purpose**: Creates Mask R-CNN model architecture
- **Base model**: ResNet-50 with Feature Pyramid Network (FPN)
- **Modifications**: Replaces classifier heads for your 105 classes

#### C. **train_mask_rcnn()** (Lines ~170-400)
- **Purpose**: Main training loop
- **What it does**:
  1. Loads dataset
  2. Creates data loaders
  3. Initializes model
  4. Sets up optimizer and learning rate scheduler
  5. Trains for N epochs
  6. Validates after each epoch
  7. Saves best model

---

### 3. **Dataset Configuration: `datasets/foodseg_pp/data.yaml`**
**Location**: `datasets/foodseg_pp/data.yaml`  
**Purpose**: Defines dataset structure  
**Contains**:
- Path to images (train/val splits)
- Path to labels (YOLO format annotations)
- Number of classes (104 food classes)
- Class names

**Example structure**:
```yaml
path: datasets/foodseg_pp
train: images/train
val: images/val
nc: 104
names: [class_0, class_1, ..., class_103]
```

---

### 4. **Dataset Files** (Not in git - too large)
**Location**: `datasets/foodseg_pp/`  
**Structure**:
```
datasets/foodseg_pp/
├── data.yaml              # Configuration (included)
├── images/
│   ├── train/             # Training images (4983 images)
│   └── val/                # Validation images (2135 images)
└── labels/
    ├── train/             # YOLO format labels (one .txt per image)
    └── val/               # YOLO format labels
```

**YOLO Label Format** (`.txt` files):
```
class_id center_x center_y width height
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.1 0.2
```
- All coordinates are normalized (0-1)
- One line per food item

---

### 5. **Inference Code: `src/detection/mask_rcnn_inference.py`**
**Location**: `src/detection/mask_rcnn_inference.py`  
**Purpose**: Use trained model for detection  
**Key class**: `MaskRCNNFoodDetector`
- Loads trained model weights
- Processes images
- Returns detections (boxes, masks, confidence, class)

**Usage**:
```python
from src.detection.mask_rcnn_inference import MaskRCNNFoodDetector

detector = MaskRCNNFoodDetector('models/mask_rcnn_food_best.pth')
results = detector.detect('meal.jpg')
```

---

## Data Flow

### Training Process:
```
1. train_mask_rcnn.py
   ↓
2. train_mask_rcnn() in src/detection/train_mask_rcnn.py
   ↓
3. FoodSegDataset loads data from datasets/foodseg_pp/
   ↓
4. DataLoader batches the data
   ↓
5. Model (Mask R-CNN) processes batches
   ↓
6. Optimizer updates model weights
   ↓
7. Best model saved to models/mask_rcnn_food_best.pth
```

### Data Format Conversion:
```
YOLO Format (input)          →    Mask R-CNN Format (output)
─────────────────────              ──────────────────────
class_id: 0-103              →    labels: 1-104 (1-indexed)
center_x, center_y          →    boxes: [x1, y1, x2, y2]
width, height (normalized)   →    boxes: absolute coordinates
(no masks)                  →    masks: rectangular masks
```

---

## Key Dependencies

### PyTorch Libraries:
- `torch` - Core PyTorch
- `torchvision` - Contains Mask R-CNN implementation
- `torchvision.models.detection.maskrcnn_resnet50_fpn` - Model architecture

### Data Processing:
- `PIL.Image` - Image loading
- `numpy` - Array operations
- `yaml` - Dataset config parsing

### Utilities:
- `tqdm` - Progress bars
- `pathlib.Path` - File path handling

---

## Output Files

### After Training:
1. **`models/mask_rcnn_food_best.pth`**
   - Trained model weights
   - Used for inference
   - ~170MB (similar to pretrained)

2. **`models/mask_rcnn_training_history.json`**
   - Training metrics per epoch
   - Loss values
   - For analysis/plotting

---

## File Summary

| File | Purpose | Size | In Git? |
|------|---------|------|---------|
| `train_mask_rcnn.py` | Entry point | Small | ✅ Yes |
| `src/detection/train_mask_rcnn.py` | Training code | ~400 lines | ✅ Yes |
| `src/detection/mask_rcnn_inference.py` | Inference code | ~100 lines | ✅ Yes |
| `datasets/foodseg_pp/data.yaml` | Dataset config | Small | ✅ Yes |
| `datasets/foodseg_pp/images/` | Training images | 1.7GB | ❌ No (too large) |
| `datasets/foodseg_pp/labels/` | YOLO labels | ~50MB | ❌ No (too large) |
| `models/mask_rcnn_food_best.pth` | Trained model | ~170MB | ✅ Yes (after training) |

---

## Quick Reference

**To train**:
```bash
python train_mask_rcnn.py
```

**To use trained model**:
```python
from src.detection.mask_rcnn_inference import MaskRCNNFoodDetector
detector = MaskRCNNFoodDetector('models/mask_rcnn_food_best.pth')
```

**Dataset location**:
- Config: `datasets/foodseg_pp/data.yaml`
- Images: `datasets/foodseg_pp/images/train/` and `val/`
- Labels: `datasets/foodseg_pp/labels/train/` and `val/`

