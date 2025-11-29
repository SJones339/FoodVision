# FoodVision
Track your Calories!

# Overview
FoodVision Nutrition AI is an AI-powered system that enables users to take a photo of a meal and instantly receive:

    - Detection of multiple foods on a single plate

    - Fine-grained classification of each food item

    - Calories + macronutrients for each detected food

    - Portion-level estimates (later milestone)

    - Seamless integration into a mobile app

The project combines YOLO-based object detection, CNN-based food classification, and USDA FoodData Central nutrition data to transform a simple meal photo into actionable dietary information.

## Main Pipeline (Version Multi-Food Detection)
Image → YOLOv8 (object detection)
     → Cropped food regions
     → Classifier (EfficientNet)
     → Food label per region
     → Nutrition pipeline (USDA)
     → Calories/macros JSON output


### Example output:

{
  "items": [
    { "name": "grilled chicken", "calories": 220 },
    { "name": "brown rice", "calories": 180 },
    { "name": "broccoli", "calories": 55 }
  ]
}

## Quick Start

**The trained models are included - you can use the pipeline immediately!**

```bash
# Install dependencies
pip install ultralytics torch efficientnet-pytorch pillow requests

# Test the pipeline
python test_pipeline.py meal.jpg
```

See [SETUP_AND_USAGE.md](SETUP_AND_USAGE.md) for detailed setup and usage instructions.

## Datasets

**Important**: Datasets are NOT included in the repository (too large for git).

**You DON'T need datasets to use the pipeline** - trained models are already included!

**You DO need datasets if you want to retrain models** (see below).

### Available Datasets

1. **Food-101** - 101 food classes, 101,000 images (for classifier training)
2. **FoodSeg103** - 104 food classes, pixel-level segmentation (for YOLO training)
3. **FoodX-251** - 251 fine-grained classes (for advanced training)

### Downloading Datasets (Optional)

Only needed if you want to retrain models:

```bash
# Download FoodSeg103 for YOLO training (1.7GB)
python download_datasets.py --foodseg103 --convert-yolo

# Download Food-101 for classifier training
python download_datasets.py --food101

# Download FoodX-251 (requires Roboflow API key)
export ROBOFLOW_API_KEY="your_key_here"
python download_datasets.py --foodx251
```

**Note**: FoodX-251 requires a Roboflow API key. Get one at https://roboflow.com/

For more details, see [SETUP_AND_USAGE.md](SETUP_AND_USAGE.md).


## Nutrition Data

The pipeline integrates with USDA FoodData Central API to provide calories and macronutrients for detected foods.

**Setup:**
1. Get a free API key: https://fdc.nal.usda.gov/api-key-signup
2. Add to `.env` file: `USDA_API_KEY=your_key_here`
3. Test with: `python test_pipeline.py meal.jpg --nutrition`

See [SETUP_AND_USAGE.md](SETUP_AND_USAGE.md) for detailed nutrition integration guide.  

## General Structure of Project :
FoodVision/  
│── data/                                   # temp downloaded images, processed crops  
│── datasets/                               # Food-101, FoodX-251, FoodSeg103  
│── src/  
│   ├── classification/                     # EfficientNet classifier code  
│   ├── detection/                          # YOLO detection pipeline  
│   ├── nutrition_pipeline/                 # USDA API + matching  
│   ├── api/                                # FastAPI backend  
│   └── utils/                              # helpers  
│── notebooks/                              # Jupyter notebooks  
│── models/                                 # saved .pt or .h5 models  
│── mobile/                                 # mobile integration (React Native/Flutter)  
│── requirements.txt  
│── README.md  

## Contributors

James — Background research, model building, mobile app

Vincent — Dataset prep, model building, integration

Stephen — Dataset collection, nutrition pipeline, API integration

