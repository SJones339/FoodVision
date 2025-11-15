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

## Datasets

We use three major datasets:

### 1. Food-101

 101 food classes

101,000 images

Good for initial classifier training


### 2. FoodX-251

251 fine-grained classes

Real-world images

Used for classifier fine-tuning


### 3. FoodSeg103 (optional, backup)

Pixel-level segmentation

Helps with multi-food plate detection


## Nutrition Data

We integrate:

USDA FoodData Central API

OpenFoodFacts API (backup)

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

