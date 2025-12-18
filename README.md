# FoodVision
Track your Calories!

# Overview
FoodVision Nutrition AI is an AI-powered system that enables users to take a photo of a meal and instantly receive:

    - Detection of multiple foods on a single plate

    - Fine-grained classification of each food item (YOLOv8 classes)

    - Calories + macronutrients for each detected food

    - Portion-level estimates (MiDaS depth + USDA serving-size fallback)

The project combines YOLOv8-seg segmentation/classification, MiDaS depth estimation, and USDA FoodData Central nutrition data to transform a meal photo into actionable dietary information.

## Main Pipeline (Version Multi-Food Detection)
Image → YOLOv8-seg (segment + classify foods)
     → MiDaS (depth map)
     → Volume estimate per segment (+ USDA serving-size fallback)
     → USDA FoodData Central (macros)
     → Calories/macros output


### Example output:

{
  "items": [
    { "name": "grilled chicken", "calories": 220 },
    { "name": "brown rice", "calories": 180 },
    { "name": "broccoli", "calories": 55 }
  ]
}

## Quick Start

```bash
# Install dependencies
python -m pip install -r requirements.txt

# Set USDA key (optional, but required for macros)
export USDA_API_KEY="your_key_here"

# Run the web app
streamlit run app.py
```

### Notes / common issues

- **First run is slow**: MiDaS may download weights the first time you upload an image.
- **If `streamlit` uses the wrong Python**: run Streamlit via Python:

```bash
python -m streamlit run app.py
```

- **No USDA key**: the app will still detect foods, but macros/totals will be blank.

## Notebook (report + experiments)

The full end-to-end experimentation and analysis is in `FinalDLFinalProject.ipynb`.

## Nutrition Data

The pipeline integrates with USDA FoodData Central API to provide calories and macronutrients for detected foods.

**Setup:**
1. Get a free API key: https://fdc.nal.usda.gov/api-key-signup
2. Add to `.env` file: `USDA_API_KEY=your_key_here`
3. Or export it in your shell before running Streamlit.

## General Structure of Project :
FoodVision/  
│── app.py                                  # Streamlit web app  
│── data/                                   # local cache/data  
│── src/  
│   └── pipeline/                           # YOLO + MiDaS + USDA pipeline  
│── models/                                 # saved .pt or .h5 models  
│── requirements.txt  
│── README.md  

## Contributors

James — Background research, model building, mobile app

Vincent — Dataset prep, model building, integration

Stephen — Dataset collection, nutrition pipeline, API integration

