# Setup and Usage Guide

Complete guide for setting up, testing, and using the FoodVision pipeline.

## Table of Contents
1. [Quick Setup](#quick-setup)
2. [Testing the Pipeline](#testing-the-pipeline)
3. [Nutrition Data Integration](#nutrition-data-integration)
4. [Dataset Setup](#dataset-setup)
5. [Troubleshooting](#troubleshooting)

---

## Quick Setup

### Prerequisites
```bash
pip install ultralytics torch efficientnet-pytorch pillow requests
```

### Models Included
- ✅ `models/yolov8m_food_best.pt` - Trained YOLOv8 detection model (ready to use!)
- ✅ `models/efficientnet_best.pth` - Trained EfficientNet classifier (ready to use!)

**Note**: You don't need to train anything - the trained models are included!

---

## Testing the Pipeline

### Basic Test

```bash
# Test with default meal.jpg
python test_pipeline.py

# Test with your own image
python test_pipeline.py path/to/your/image.jpg
```

### Test with Nutrition Data

```bash
# Requires USDA API key (see Nutrition section)
python test_pipeline.py meal.jpg --nutrition
```

### What You'll Get

**Console Output:**
- Number of food items detected
- Food names and confidence scores
- Nutrition data (if enabled)

**Output Files:**
- `models/pipeline_result.jpg` - Visualization with bounding boxes
- `data/processed/crops/` - Individual food crops
- `models/pipeline_results.json` - Full results in JSON

### Understanding Results

**Detection Confidence:**
- **>0.8**: Excellent detection
- **0.6-0.8**: Good detection
- **<0.6**: May be uncertain

**Classification Confidence:**
- **>0.8**: Very confident classification
- **0.6-0.8**: Reasonable confidence
- **<0.6**: Less certain (check top predictions)

### Compare Models

```bash
# With trained model (default)
python test_pipeline.py meal.jpg

# With pretrained model (for comparison)
python test_pipeline.py meal.jpg --pretrained
```

---

## Nutrition Data Integration

### Setup USDA API Key

1. **Get free API key**: https://fdc.nal.usda.gov/api-guide.html

2. **Add to `.env` file** (recommended):
   ```bash
   USDA_API_KEY=your_key_here
   ```

3. **Or set environment variable**:
   ```bash
   export USDA_API_KEY='your_key_here'
   ```

### How It Works

The nutrition pipeline automatically:
- Maps Food-101 class names to USDA search terms
- Searches USDA database for nutrition data
- Caches results for performance
- Returns calories, protein, carbs, fat, and more

### Food Name Mapping

Food-101 uses underscores (e.g., `apple_pie`), but USDA needs natural language (e.g., `apple pie`). The system handles this automatically:

| Food-101 Name | USDA Search Term |
|---------------|------------------|
| `apple_pie` | `apple pie` |
| `chicken_curry` | `chicken curry` |
| `macaroni_and_cheese` | `macaroni and cheese` |
| `grilled_salmon` | `salmon grilled` |

### Usage in Code

```python
from src.pipeline.end_to_end import FoodVisionPipeline

# Initialize with nutrition
pipeline = FoodVisionPipeline(
    detection_model_path='models/yolov8m_food_best.pt',
    classification_model_path='models/efficientnet_best.pth',
    include_nutrition=True
)

# Process image
results = pipeline.process_image('meal.jpg', include_nutrition=True)

# Access nutrition data
for item in results['items']:
    print(f"{item['food_name']}: {item.get('calories', 'N/A')} calories")
    if item.get('nutrition'):
        print(f"  Protein: {item['nutrition']['protein']}g")
        print(f"  Carbs: {item['nutrition']['carbs']}g")
        print(f"  Fat: {item['nutrition']['fat']}g")

# Total nutrition
summary = results['nutrition_summary']
print(f"Total Calories: {summary['total_calories']}")
```

### Standalone Nutrition API

```python
from src.nutrition_pipeline import NutritionAPI

api = NutritionAPI()

# Search for nutrition data
nutrition = api.search_food('apple_pie')

if nutrition:
    print(f"Calories: {nutrition['calories']}")
    print(f"Protein: {nutrition['protein']}g")
    print(f"Carbs: {nutrition['carbs']}g")
    print(f"Fat: {nutrition['fat']}g")
```

### Caching

Nutrition results are automatically cached in `data/nutrition_cache.json` to:
- Avoid redundant API calls
- Improve performance
- Work offline for previously searched foods

---

## Dataset Setup

### Important: Datasets Not in Git

**The following datasets are NOT included in the repository** (too large for git):
- `datasets/foodseg_pp/` - 1.7GB (YOLO training dataset)
- `datasets/food101/` - Large dataset
- `datasets/foodseg103/` - Large dataset
- `datasets/foodx251/` - Large dataset

### Do You Need the Datasets?

**You DON'T need datasets if you're just using the pipeline:**
- ✅ Trained models are already included
- ✅ You can test and use the pipeline immediately
- ✅ No training required

**You DO need datasets if you want to:**
- 🔄 Retrain YOLOv8
- 🔄 Retrain EfficientNet
- 🔄 Test on validation images
- 🔄 Evaluate model performance

### Downloading Datasets

If you need datasets for training or evaluation:

```bash
# Download all datasets
python download_datasets.py --all

# Download specific datasets
python download_datasets.py --foodseg103 --convert-yolo  # For YOLO training
python download_datasets.py --food101                    # For classifier training
python download_datasets.py --foodx251                   # Requires Roboflow API key
```

### FoodSeg103 (for YOLO Training)

**To download and convert FoodSeg103 for YOLO training:**

```bash
# Download and convert to YOLO format
python download_datasets.py --foodseg103 --convert-yolo
```

This will:
1. Download FoodSeg103 dataset
2. Convert to YOLO format
3. Save to `datasets/foodseg_pp/`
4. Create `datasets/foodseg_pp/data.yaml` for training

**Note**: This is a large download (~1.7GB). Only needed if you want to retrain YOLOv8.

### Dataset Locations

After downloading:
- `datasets/foodseg_pp/` - YOLO format dataset (for detection training)
- `datasets/food101/` - Food-101 dataset (for classification training)
- `datasets/foodseg103/` - Original FoodSeg103 (if needed)
- `datasets/foodx251/` - FoodX-251 dataset (if downloaded)

---

## Troubleshooting

### "No food detected"
- Try a clearer image with better lighting
- Make sure foods are visible and not too small
- Foods should be in FoodSeg103 classes (104 food types)

### "Wrong classifications"
- YOLOv8 is trained on FoodSeg103 (104 classes)
- EfficientNet is trained on Food-101 (101 classes)
- Some foods might not match perfectly
- Check "top_predictions" for alternatives

### "Model not found"
- Make sure `models/yolov8m_food_best.pt` exists (~52MB)
- Make sure `models/efficientnet_best.pth` exists
- Check file paths in your code

### "USDA API key not found"
- Create `.env` file with `USDA_API_KEY=your_key`
- Or set environment variable: `export USDA_API_KEY='your_key'`
- Get free key: https://fdc.nal.usda.gov/api-guide.html

### "No nutrition data found"
- Some foods might not be in USDA database
- Try different search terms
- Check if food name mapping is correct

### Import Errors
```bash
# Install missing packages
pip install ultralytics efficientnet-pytorch torch pillow requests
```

### Device Issues
- The pipeline auto-detects device (CUDA/MPS/CPU)
- MPS (Apple Silicon) is automatically used if available
- Check device in console output

---

## Advanced Usage

### Programmatic Usage

```python
from src.pipeline.end_to_end import FoodVisionPipeline

# Initialize pipeline
pipeline = FoodVisionPipeline(
    detection_model_path='models/yolov8m_food_best.pt',
    classification_model_path='models/efficientnet_best.pth',
    include_nutrition=True
)

# Process multiple images
images = ['meal1.jpg', 'meal2.jpg', 'meal3.jpg']
for img in images:
    results = pipeline.process_image(img)
    print(f"{img}: {results['num_detections']} items detected")
```

### Custom Confidence Threshold

```python
pipeline = FoodVisionPipeline(
    detection_model_path='models/yolov8m_food_best.pt',
    conf_threshold=0.3  # Lower = more detections (may include false positives)
)
```

### Save Cropped Regions

```python
results = pipeline.process_image('meal.jpg', save_crops=True)
# Crops saved to: data/processed/crops/
```

### Get Full Results

```python
results = pipeline.process_image('meal.jpg', include_nutrition=True)

# Access all data
for item in results['items']:
    print(f"Item {item['item_id']}: {item['food_name']}")
    print(f"  BBox: {item['bbox']}")
    print(f"  Detection: {item['detection_confidence']:.2f}")
    print(f"  Classification: {item['classification_confidence']:.2f}")
    print(f"  Calories: {item.get('calories', 'N/A')}")
    print(f"  Top predictions: {item['top_predictions']}")
```

---

## Tips for Best Results

1. **Good lighting**: Clear, well-lit photos work best
2. **Multiple foods**: Pipeline is designed for multi-food detection
3. **Clear view**: Foods should be clearly visible, not obscured
4. **Food classes**: Works best for foods in FoodSeg103 (104 classes) and Food-101 (101 classes)
5. **Image quality**: Higher resolution images generally work better

---

## Model Information

### YOLOv8 Detection Model
- **File**: `models/yolov8m_food_best.pt`
- **Size**: ~52MB
- **Training**: FoodSeg103 dataset
- **Classes**: 104 food classes
- **Purpose**: Detects and segments food regions

### EfficientNet Classifier
- **File**: `models/efficientnet_best.pth`
- **Training**: Food-101 dataset
- **Classes**: 101 food classes
- **Purpose**: Classifies detected food regions

---

For more information, see the main [README.md](README.md).

