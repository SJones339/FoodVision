# Trained Models

This directory contains the trained models for the FoodVision project.

## Models Included

- **`custom_cnn_best.pth`** (~16MB)
  - Custom CNN trained from scratch on Food-101
  - Validation Accuracy: ~42%
  - Trained for demonstration/comparison purposes

- **`efficientnet_best.pth`** (~16MB)
  - EfficientNet-B0 with transfer learning (ImageNet → Food-101)
  - Validation Accuracy: ~85%
  - **This is the production model** - use this for the pipeline

- **`training_results.json`**
  - Training metrics and comparison data

## Usage

The models are automatically loaded by the pipeline:

```python
from pipeline.end_to_end import FoodVisionPipeline

pipeline = FoodVisionPipeline(
    classification_model_path='models/efficientnet_best.pth'
)
```

## Note

These models are included in the repository so teammates don't need to retrain.
Total size: ~32MB (acceptable for GitHub).

