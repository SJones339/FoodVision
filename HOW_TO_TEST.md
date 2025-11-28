# How to Test the Food Vision Pipeline

## Quick Start

### Step 1: Get a Test Image
1. Find or take a photo of a meal with multiple foods
2. Save it to your project folder (e.g., `meal.jpg` or `test_image.jpg`)
3. Make sure it's a clear photo with visible food items

### Step 2: Run the Test
```bash
python test_pipeline.py meal.jpg
```

Or if you named it `test_image.jpg` or `meal.jpg`:
```bash
python test_pipeline.py
```

### Step 3: Check Results
- **Console output**: Shows detected foods and classifications
- **Visualization**: `models/pipeline_result.jpg` - Image with bounding boxes and labels
- **Cropped regions**: `data/processed/crops/` - Individual food crops

## Detailed Example

```bash
# 1. Place your image in the project root
#    Example: /Users/stephenjones/Desktop/FoodVision/my_meal.jpg

# 2. Run the test
python test_pipeline.py my_meal.jpg

# 3. Output you'll see:
#    - Processing steps (detection, classification)
#    - List of detected foods with confidence scores
#    - Visualization saved to models/pipeline_result.jpg
```

## What the Pipeline Does

1. **YOLOv8 Detection**: Finds food regions in your image
2. **Crop Regions**: Extracts each detected food item
3. **EfficientNet Classification**: Identifies what each food is
4. **Output**: 
   - JSON with all results
   - Visualized image with bounding boxes
   - Individual food crops

## Expected Output

```
FOOD VISION PIPELINE TEST
============================================================
Using device: mps
Loading EfficientNet from models/efficientnet_best.pth
✓ Pipeline initialized!

Processing image: my_meal.jpg
============================================================
Step 1: Detecting food regions...
  Found 3 food regions

Step 2: Classifying food items...
  Item 1: grilled_chicken (det: 0.85, cls: 0.92)
  Item 2: fried_rice (det: 0.78, cls: 0.88)
  Item 3: broccoli (det: 0.72, cls: 0.85)

============================================================
✓ Processed 3 food items

RESULTS
============================================================
{
  "image_path": "my_meal.jpg",
  "num_detections": 3,
  "items": [
    {
      "item_id": 1,
      "bbox": [100, 150, 300, 350],
      "detection_confidence": 0.85,
      "food_name": "grilled_chicken",
      "classification_confidence": 0.92
    },
    ...
  ]
}

✓ Results saved to models/pipeline_result.jpg
✓ Cropped regions saved to data/processed/crops/
```

## Troubleshooting

**No food detected?**
- Try a clearer image
- Make sure foods are visible and not too small
- Adjust confidence threshold (in code)

**Wrong classifications?**
- Normal - model is trained on Food-101 (101 classes)
- Some foods might not be in the training set
- Confidence scores show how certain the model is

**Error loading model?**
- Make sure `models/efficientnet_best.pth` exists
- Check that you've completed training

## Tips for Best Results

1. **Good lighting**: Clear, well-lit photos work best
2. **Multiple foods**: The pipeline is designed for multi-food detection
3. **Clear view**: Foods should be clearly visible, not obscured
4. **Food-101 classes**: Works best for foods in Food-101 dataset (101 common foods)

