# FoodVision: Deep Learning Sprint Plan (2-3 Days)

## Mission Statement

**Primary Goal**: Develop an AI-powered system that uses deep learning computer vision techniques to automatically identify foods from meal images and estimate their nutritional content, addressing the global obesity epidemic by making calorie tracking effortless and accurate.

**Core Deep Learning Components**:
1. **Object Detection & Segmentation**: YOLOv8-based model to detect and segment multiple food items in a single meal image
2. **Image Classification**: EfficientNet-based CNN to classify detected food regions into specific food categories
3. **Multi-Stage Pipeline**: Combining detection → classification → nutrition mapping using deep learning models

**Problem**: Manual calorie logging is tedious and inaccurate, leading to poor dietary awareness. Our solution uses state-of-the-art deep learning models to automate food recognition and nutrition estimation from photos.

**Why Deep Learning**: Traditional computer vision fails at the complexity of food recognition (variations in preparation, lighting, angles, multiple items). Deep CNNs and modern architectures (YOLO, EfficientNet) can learn these complex patterns from large-scale datasets.

---

## Architecture: Why Two Models? (YOLOv8 + EfficientNet)

### The Question: "If YOLOv8 can classify, why EfficientNet?"

**Answer**: Each model is optimized for different tasks. YOLOv8 excels at detection (WHERE food is), while EfficientNet excels at fine-grained classification (WHAT food it is).

### YOLOv8's Role:
- **Primary**: Detection & segmentation (finds food regions)
- **Secondary**: Rough classification (~40-50% accuracy on Food-101)
- **Limitation**: Classification head is not optimized (detection is priority)
- **Classes**: 103 FoodSeg103 classes (different from Food-101)

### EfficientNet's Role:
- **Primary**: Fine-grained classification (identifies specific foods)
- **Accuracy**: 85% on Food-101 (2x better than YOLOv8's classification)
- **Optimization**: Entire architecture designed for classification
- **Classes**: 101 Food-101 classes (more fine-grained)

### Why Both?
- **YOLOv8**: Best at finding WHERE food is (detection)
- **EfficientNet**: Best at identifying WHAT food is (classification)
- **Together**: Best detection + Best classification = Best overall system

**See `ARCHITECTURE_EXPLANATION.md` for detailed explanation.**

---

## Current Status Assessment

### What We Have:
- ✅ **Baseline YOLOv8n-seg model** trained for 5 epochs
- ✅ **Data preprocessing pipeline** (FoodSeg103 → YOLO format conversion)
- ✅ **Dataset access** (FoodSeg103, Food-101, FoodX-251)
- ✅ **Training infrastructure** (notebooks with training code)

### Current Model Performance:
- **Segmentation mAP50**: 0.2417 (24.17%) - **Needs 2x improvement**
- **Segmentation mAP50-95**: 0.1839 (18.39%)
- **Precision**: 0.4958 (49.58%) - **Needs 1.4x improvement**
- **Recall**: 0.2537 (25.37%) - **Needs 2.4x improvement** ⚠️ Critical
- **F1-score**: 0.2482 (24.82%)

**Verdict**: Baseline demonstrates pipeline works, but model is severely undertrained and using smallest architecture variant.

---

## 2-3 Day Deep Learning Sprint Plan

### Day 1: Detection Model Improvement & Training (8-10 hours)

#### Morning (3-4 hours): Model Architecture & Training Setup

**Task 1.1: Upgrade Model Architecture** (30 min)
- **Action**: Switch from YOLOv8n-seg to YOLOv8s-seg or YOLOv8m-seg
- **Why**: Larger models have more capacity for complex food segmentation
- **Code Change**:
  ```python
  # notebooks/02_yolo_model.ipynb
  # Change from:
  model = YOLO("yolov8n-seg.pt")
  # To:
  model = YOLO("yolov8s-seg.pt")  # or yolov8m-seg.pt
  ```
- **Expected Impact**: +10-15% mAP improvement

**Task 1.2: Implement Data Augmentation** (45 min)
- **Action**: Add augmentation pipeline to improve generalization
- **Why**: Food images vary in lighting, angle, background - augmentation teaches robustness
- **Code**:
  ```python
  model.train(
      data="data/foodseg_pp/data.yaml",
      epochs=50,  # Increase from 5
      imgsz=640,
      batch=16,  # Increase if GPU allows
      device=0,
      augment=True,  # Enable augmentation
      hsv_h=0.015,  # Hue augmentation
      hsv_s=0.7,    # Saturation augmentation
      hsv_v=0.4,    # Value augmentation
      degrees=10,   # Rotation
      translate=0.1,
      scale=0.5,    # Scaling
      flipud=0.0,   # No vertical flip (food orientation matters)
      fliplr=0.5,   # Horizontal flip OK
      mosaic=1.0,   # Mosaic augmentation
      mixup=0.1,    # Mixup augmentation
  )
  ```
- **Expected Impact**: +5-10% mAP improvement, better generalization

**Task 1.3: Extended Training Configuration** (30 min)
- **Action**: Set up proper training with early stopping and learning rate scheduling
- **Why**: 5 epochs is insufficient - need 50-100 epochs with proper regularization
- **Code**:
  ```python
  model.train(
      data="data/foodseg_pp/data.yaml",
      epochs=100,  # Extended training
      patience=15,  # Early stopping if no improvement
      imgsz=640,
      batch=16,
      device=0,
      workers=4,
      optimizer='AdamW',  # Better optimizer
      lr0=0.001,  # Initial learning rate
      lrf=0.01,   # Final learning rate (1% of initial)
      momentum=0.937,
      weight_decay=0.0005,
      warmup_epochs=3,  # Learning rate warmup
      warmup_momentum=0.8,
      box=7.5,    # Box loss gain
      cls=0.5,    # Class loss gain
      dfl=1.5,    # DFL loss gain
  )
  ```
- **Expected Impact**: +20-30% mAP improvement from proper training

**Task 1.4: Create Training Script** (1 hour)
- **Action**: Convert notebook to reusable Python script
- **Why**: Better for production, easier to run multiple experiments
- **File**: `src/detection/train_yolo.py`
- **Code Structure**:
  ```python
  from ultralytics import YOLO
  import argparse
  from pathlib import Path
  
  def train_detection_model(
      model_size='s',  # 'n', 's', 'm', 'l', 'x'
      epochs=100,
      batch_size=16,
      img_size=640,
      data_yaml='data/foodseg_pp/data.yaml',
      output_dir='models/'
  ):
      model = YOLO(f"yolov8{model_size}-seg.pt")
      
      results = model.train(
          data=data_yaml,
          epochs=epochs,
          imgsz=img_size,
          batch=batch_size,
          device=0,
          # ... all hyperparameters
      )
      
      # Save best model
      best_model = results.save_dir / 'weights' / 'best.pt'
      return best_model
  ```

#### Afternoon (4-5 hours): Training Execution & Monitoring

**Task 1.5: Start Extended Training** (4-5 hours - mostly waiting)
- **Action**: Launch training job on GPU (Colab or local)
- **Monitoring**: 
  - Watch loss curves (should decrease steadily)
  - Monitor validation mAP (should increase)
  - Check for overfitting (val loss increasing while train loss decreases)
- **Checkpoints**: Model saves automatically every epoch
- **Expected Training Time**: 4-6 hours for 50-100 epochs on GPU

**Task 1.6: Model Evaluation & Analysis** (1 hour)
- **Action**: Evaluate trained model on validation set
- **Metrics to Track**:
  ```python
  from ultralytics import YOLO
  
  model = YOLO('models/best.pt')
  metrics = model.val(data='data/foodseg_pp/data.yaml')
  
  print(f"mAP50: {metrics.seg.map50:.4f}")
  print(f"mAP50-95: {metrics.seg.map:.4f}")
  print(f"Precision: {metrics.seg.p.mean():.4f}")
  print(f"Recall: {metrics.seg.r.mean():.4f}")
  ```
- **Analysis**:
  - Compare to baseline (should see 2x improvement)
  - Visualize predictions on sample images
  - Identify failure cases (what foods are missed?)

**Task 1.7: Hyperparameter Experimentation** (1 hour, parallel with training)
- **Action**: Test different configurations
- **Experiments**:
  - Different image sizes: 640, 800, 1280
  - Different batch sizes: 8, 16, 32
  - Different learning rates: 0.001, 0.0005, 0.002
- **Document**: Keep log of experiments and results

#### Evening (1-2 hours): Detection Pipeline Integration

**Task 1.8: Create Detection Inference Script** (1 hour)
- **Action**: Build reusable inference pipeline
- **File**: `src/detection/inference.py`
- **Code**:
  ```python
  from ultralytics import YOLO
  from PIL import Image
  import numpy as np
  
  class FoodDetector:
      def __init__(self, model_path='models/best.pt'):
          self.model = YOLO(model_path)
      
      def detect(self, image_path, conf_threshold=0.25):
          """
          Detect and segment food items in image.
          
          Returns:
              List of detections with:
              - bbox: bounding box coordinates
              - mask: segmentation mask
              - confidence: detection confidence
              - class_id: predicted class
          """
          results = self.model.predict(
              source=image_path,
              conf=conf_threshold,
              imgsz=640,
              save=False
          )
          
          detections = []
          for result in results:
              boxes = result.boxes
              masks = result.masks
              
              for i in range(len(boxes)):
                  detection = {
                      'bbox': boxes.xyxy[i].cpu().numpy(),
                      'confidence': boxes.conf[i].cpu().numpy(),
                      'class_id': int(boxes.cls[i].cpu().numpy()),
                      'mask': masks.data[i].cpu().numpy() if masks else None
                  }
                  detections.append(detection)
          
          return detections
  ```

**Day 1 Deliverables**:
- ✅ Improved YOLOv8 model (YOLOv8s or YOLOv8m)
- ✅ Trained model with 50-100 epochs
- ✅ Evaluation metrics showing improvement
- ✅ Detection inference pipeline
- ✅ Training script (`src/detection/train_yolo.py`)

**Expected Results**: mAP50 > 0.40, Recall > 0.40 (2x improvement from baseline)

---

### Day 2: Classification Model & Integration (8-10 hours)

#### Morning (3-4 hours): EfficientNet Classification Setup

**Task 2.1: Prepare Classification Dataset** (1 hour)
- **Action**: Set up Food-101 for EfficientNet training
- **Why**: Food-101 has 101 classes, 101k images - perfect for classification
- **Code**:
  ```python
  # notebooks/03_classification_setup.ipynb
  from datasets import load_dataset
  from pathlib import Path
  from PIL import Image
  import torch
  from torch.utils.data import Dataset, DataLoader
  from torchvision import transforms
  
  # Load Food-101
  ds = load_dataset("ethz/food101")
  
  # Create PyTorch dataset
  class Food101Dataset(Dataset):
      def __init__(self, split='train', transform=None):
          self.data = ds[split]
          self.transform = transform
          
      def __len__(self):
          return len(self.data)
      
      def __getitem__(self, idx):
          item = self.data[idx]
          image = item['image']
          label = item['label']
          
          if self.transform:
              image = self.transform(image)
          
          return image, label
  ```

**Task 2.2: EfficientNet Model Setup** (1 hour)
- **Action**: Implement EfficientNet-B0 or B1 for food classification
- **Why**: EfficientNet balances accuracy and efficiency (good for mobile later)
- **Code**:
  ```python
  # src/classification/model.py
  import torch
  import torch.nn as nn
  from efficientnet_pytorch import EfficientNet
  
  class FoodClassifier(nn.Module):
      def __init__(self, num_classes=101, model_name='efficientnet-b0'):
          super().__init__()
          # Load pretrained EfficientNet
          self.backbone = EfficientNet.from_pretrained(model_name)
          
          # Replace classifier head
          num_features = self.backbone._fc.in_features
          self.backbone._fc = nn.Linear(num_features, num_classes)
          
      def forward(self, x):
          return self.backbone(x)
  ```

**Task 2.3: Training Configuration** (1 hour)
- **Action**: Set up training loop with proper data augmentation
- **Code**:
  ```python
  # src/classification/train.py
  import torch
  import torch.nn as nn
  import torch.optim as optim
  from torch.utils.data import DataLoader
  from torchvision import transforms
  from tqdm import tqdm
  
  def train_classifier(
      model,
      train_loader,
      val_loader,
      epochs=30,
      lr=0.001,
      device='cuda'
  ):
      criterion = nn.CrossEntropyLoss()
      optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
      scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
      
      best_acc = 0.0
      
      for epoch in range(epochs):
          # Training
          model.train()
          train_loss = 0.0
          correct = 0
          total = 0
          
          for images, labels in tqdm(train_loader):
              images, labels = images.to(device), labels.to(device)
              
              optimizer.zero_grad()
              outputs = model(images)
              loss = criterion(outputs, labels)
              loss.backward()
              optimizer.step()
              
              train_loss += loss.item()
              _, predicted = outputs.max(1)
              total += labels.size(0)
              correct += predicted.eq(labels).sum().item()
          
          # Validation
          model.eval()
          val_loss = 0.0
          val_correct = 0
          val_total = 0
          
          with torch.no_grad():
              for images, labels in val_loader:
                  images, labels = images.to(device), labels.to(device)
                  outputs = model(images)
                  loss = criterion(outputs, labels)
                  
                  val_loss += loss.item()
                  _, predicted = outputs.max(1)
                  val_total += labels.size(0)
                  val_correct += predicted.eq(labels).sum().item()
          
          train_acc = 100. * correct / total
          val_acc = 100. * val_correct / val_total
          
          print(f'Epoch {epoch+1}/{epochs}:')
          print(f'  Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%')
          print(f'  Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.2f}%')
          
          scheduler.step()
          
          # Save best model
          if val_acc > best_acc:
              best_acc = val_acc
              torch.save(model.state_dict(), 'models/efficientnet_food101_best.pth')
      
      return model
  ```

**Task 2.4: Data Augmentation for Classification** (30 min)
- **Action**: Define augmentation pipeline
- **Code**:
  ```python
  # Training augmentations
  train_transform = transforms.Compose([
      transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
      transforms.RandomHorizontalFlip(),
      transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
      transforms.RandomRotation(15),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225])
  ])
  
  # Validation (no augmentation)
  val_transform = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225])
  ])
  ```

#### Afternoon (4-5 hours): Classification Training & Fine-tuning

**Task 2.5: Train EfficientNet on Food-101** (3-4 hours - mostly waiting)
- **Action**: Launch training on Food-101 dataset
- **Expected Results**:
  - Training accuracy: > 90% after 30 epochs
  - Validation accuracy: > 85%
  - Training time: 2-3 hours on GPU

**Task 2.6: Fine-tune on FoodX-251** (1-2 hours)
- **Action**: Fine-tune trained model on FoodX-251 for finer-grained classification
- **Why**: FoodX-251 has 251 classes - more specific food categories
- **Code**:
  ```python
  # Load Food-101 trained model
  model = FoodClassifier(num_classes=101)
  model.load_state_dict(torch.load('models/efficientnet_food101_best.pth'))
  
  # Replace classifier head for 251 classes
  num_features = model.backbone._fc.in_features
  model.backbone._fc = nn.Linear(num_features, 251)
  
  # Fine-tune with lower learning rate
  optimizer = optim.AdamW(model.parameters(), lr=0.0001)  # 10x lower LR
  # Train for 10-15 epochs
  ```

**Task 2.7: Create Classification Inference** (1 hour)
- **Action**: Build inference pipeline for classification
- **File**: `src/classification/inference.py`
- **Code**:
  ```python
  import torch
  from PIL import Image
  from torchvision import transforms
  import json
  
  class FoodClassifier:
      def __init__(self, model_path, class_names_path, device='cuda'):
          self.device = device
          self.model = FoodClassifier(num_classes=251)
          self.model.load_state_dict(torch.load(model_path))
          self.model.to(device)
          self.model.eval()
          
          # Load class names
          with open(class_names_path) as f:
              self.class_names = json.load(f)
          
          self.transform = transforms.Compose([
              transforms.Resize(256),
              transforms.CenterCrop(224),
              transforms.ToTensor(),
              transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
          ])
      
      def classify(self, image_path, top_k=5):
          """
          Classify food image.
          
          Returns:
              List of (class_name, confidence) tuples
          """
          image = Image.open(image_path).convert('RGB')
          image_tensor = self.transform(image).unsqueeze(0).to(self.device)
          
          with torch.no_grad():
              outputs = self.model(image_tensor)
              probabilities = torch.softmax(outputs, dim=1)
              top_probs, top_indices = torch.topk(probabilities, top_k)
          
          results = []
          for prob, idx in zip(top_probs[0], top_indices[0]):
              results.append({
                  'class': self.class_names[idx.item()],
                  'confidence': prob.item()
              })
          
          return results
  ```

#### Evening (1-2 hours): Pipeline Integration

**Task 2.8: Integrate Detection + Classification** (1-2 hours)
- **Action**: Combine YOLO detection with EfficientNet classification
- **File**: `src/pipeline/end_to_end.py`
- **Code**:
  ```python
  from detection.inference import FoodDetector
  from classification.inference import FoodClassifier
  from PIL import Image
  import numpy as np
  
  class FoodVisionPipeline:
      def __init__(self, detection_model_path, classification_model_path):
          self.detector = FoodDetector(detection_model_path)
          self.classifier = FoodClassifier(classification_model_path)
      
      def process_image(self, image_path):
          """
          Full pipeline: Detection → Classification → Results
          """
          # Step 1: Detect food regions
          detections = self.detector.detect(image_path)
          
          # Step 2: Classify each detected region
          image = Image.open(image_path)
          results = []
          
          for det in detections:
              # Crop region from image
              bbox = det['bbox']
              x1, y1, x2, y2 = map(int, bbox)
              crop = image.crop((x1, y1, x2, y2))
              
              # Save crop temporarily
              crop_path = 'temp_crop.jpg'
              crop.save(crop_path)
              
              # Classify crop
              classifications = self.classifier.classify(crop_path, top_k=1)
              
              results.append({
                  'bbox': bbox,
                  'mask': det['mask'],
                  'detection_confidence': det['confidence'],
                  'food_name': classifications[0]['class'],
                  'classification_confidence': classifications[0]['confidence']
              })
          
          return results
  ```

**Day 2 Deliverables**:
- ✅ EfficientNet model trained on Food-101
- ✅ Fine-tuned on FoodX-251
- ✅ Classification accuracy > 85%
- ✅ Integrated detection + classification pipeline
- ✅ End-to-end inference script

**Expected Results**: Classification accuracy > 85% on Food-101, > 75% on FoodX-251

---

### Day 3: Nutrition Pipeline, Evaluation & Presentation Prep (8-10 hours)

#### Morning (3-4 hours): Nutrition Integration & Final Pipeline

**Task 3.1: USDA API Integration** (2 hours)
- **Action**: Connect classification results to nutrition data
- **File**: `src/nutrition_pipeline/nutrition_api.py`
- **Code**:
  ```python
  import requests
  import json
  from typing import Dict, Optional
  
  class NutritionAPI:
      def __init__(self, api_key):
          self.api_key = api_key
          self.base_url = "https://api.nal.usda.gov/fdc/v1"
          self.cache = {}  # Simple cache
      
      def search_food(self, food_name: str) -> Optional[Dict]:
          """
          Search USDA database for food.
          Returns nutrition data if found.
          """
          if food_name in self.cache:
              return self.cache[food_name]
          
          # Normalize food name for search
          search_term = self._normalize_food_name(food_name)
          
          # Search API
          url = f"{self.base_url}/foods/search"
          params = {
              'api_key': self.api_key,
              'query': search_term,
              'pageSize': 1
          }
          
          try:
              response = requests.get(url, params=params)
              data = response.json()
              
              if data.get('foods'):
                  food = data['foods'][0]
                  nutrition = self._extract_nutrition(food)
                  self.cache[food_name] = nutrition
                  return nutrition
          except Exception as e:
              print(f"Error fetching nutrition: {e}")
          
          return None
      
      def _normalize_food_name(self, name: str) -> str:
          """Map FoodX-251 class names to USDA search terms"""
          # Simple mapping - can be expanded
          mappings = {
              'grilled_chicken': 'chicken breast grilled',
              'fried_rice': 'fried rice',
              'pasta_carbonara': 'pasta carbonara',
              # ... more mappings
          }
          return mappings.get(name.lower(), name)
      
      def _extract_nutrition(self, food_data: Dict) -> Dict:
          """Extract key nutrition info from USDA response"""
          nutrients = {}
          for nutrient in food_data.get('foodNutrients', []):
              name = nutrient.get('nutrientName', '').lower()
              value = nutrient.get('value', 0)
              
              if 'energy' in name or 'calories' in name:
                  nutrients['calories'] = value
              elif 'protein' in name:
                  nutrients['protein'] = value
              elif 'carbohydrate' in name:
                  nutrients['carbs'] = value
              elif 'fat' in name and 'total' in name:
                  nutrients['fat'] = value
          
          return nutrients
  ```

**Task 3.2: Complete End-to-End Pipeline** (1 hour)
- **Action**: Integrate nutrition API into full pipeline
- **File**: `src/pipeline/complete_pipeline.py`
- **Code**:
  ```python
  from pipeline.end_to_end import FoodVisionPipeline
  from nutrition_pipeline.nutrition_api import NutritionAPI
  
  class CompleteFoodVisionSystem:
      def __init__(self, detection_model, classification_model, usda_api_key):
          self.pipeline = FoodVisionPipeline(detection_model, classification_model)
          self.nutrition_api = NutritionAPI(usda_api_key)
      
      def analyze_meal(self, image_path):
          """
          Complete analysis: Detection → Classification → Nutrition
          
          Returns:
              {
                  'items': [
                      {
                          'food_name': 'grilled chicken',
                          'confidence': 0.92,
                          'calories': 220,
                          'protein': 27.0,
                          'carbs': 0.0,
                          'fat': 11.0,
                          'bbox': [...],
                          'mask': [...]
                      }
                  ],
                  'total_calories': 455
              }
          """
          # Get detections and classifications
          detections = self.pipeline.process_image(image_path)
          
          # Get nutrition for each item
          results = []
          total_calories = 0
          
          for det in detections:
              food_name = det['food_name']
              nutrition = self.nutrition_api.search_food(food_name)
              
              item = {
                  'food_name': food_name,
                  'detection_confidence': det['detection_confidence'],
                  'classification_confidence': det['classification_confidence'],
                  'bbox': det['bbox'],
                  'mask': det['mask']
              }
              
              if nutrition:
                  item.update(nutrition)
                  total_calories += nutrition.get('calories', 0)
              else:
                  item['calories'] = None
                  item['protein'] = None
                  item['carbs'] = None
                  item['fat'] = None
              
              results.append(item)
          
          return {
              'items': results,
              'total_calories': total_calories,
              'num_items': len(results)
          }
  ```

**Task 3.3: Create Evaluation Script** (1 hour)
- **Action**: Build comprehensive evaluation on test set
- **File**: `src/evaluation/evaluate_pipeline.py`
- **Metrics to Compute**:
  ```python
  def evaluate_pipeline(pipeline, test_images, ground_truth):
      """
      Evaluate complete pipeline:
      1. Detection metrics (mAP, precision, recall)
      2. Classification accuracy
      3. End-to-end accuracy (correct food + nutrition)
      """
      detection_metrics = evaluate_detection(pipeline.detector, test_images)
      classification_metrics = evaluate_classification(pipeline.classifier, test_images)
      end_to_end_metrics = evaluate_end_to_end(pipeline, test_images, ground_truth)
      
      return {
          'detection': detection_metrics,
          'classification': classification_metrics,
          'end_to_end': end_to_end_metrics
      }
  ```

#### Afternoon (3-4 hours): Deep Learning Analysis & Visualization

**Task 3.4: Model Analysis & Interpretability** (2 hours)
- **Action**: Analyze what the models learned
- **Techniques to Demonstrate**:
  1. **Grad-CAM Visualization**: Show what parts of image model focuses on
  2. **Confusion Matrix**: Identify which foods are confused
  3. **Failure Case Analysis**: Why certain foods fail
  4. **Feature Visualization**: What features the CNN learned

**Code for Grad-CAM**:
```python
# src/analysis/visualize_attention.py
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image

def generate_gradcam(model, image, target_class):
    """
    Generate Grad-CAM visualization showing where model focuses.
    """
    model.eval()
    image.requires_grad = True
    
    # Forward pass
    output = model(image)
    
    # Backward pass
    output[0, target_class].backward()
    
    # Get gradients
    gradients = image.grad.data
    activations = model.get_activations(image)
    
    # Weight gradients by activations
    weights = torch.mean(gradients, dim=(2, 3))
    cam = torch.zeros(activations.shape[2:])
    
    for i, w in enumerate(weights[0]):
        cam += w * activations[0, i, :, :]
    
    # Normalize and overlay
    cam = F.relu(cam)
    cam = cam / cam.max()
    cam = cam.cpu().numpy()
    
    return cam
```

**Task 3.5: Create Visualization Scripts** (1 hour)
- **Action**: Visualize predictions, attention maps, metrics
- **Outputs**:
  - Side-by-side: Original image vs. predictions with bounding boxes
  - Segmentation masks overlaid on images
  - Attention heatmaps (Grad-CAM)
  - Training curves (loss, accuracy over epochs)
  - Confusion matrices

**Task 3.6: Performance Benchmarks** (1 hour)
- **Action**: Measure model performance
- **Metrics**:
  - Inference time per image
  - Memory usage
  - Model size
  - Accuracy vs. speed trade-offs
- **Document**: Create performance report

#### Evening (2-3 hours): Presentation Preparation

**Task 3.7: Create Presentation Notebook** (2 hours)
- **Action**: Jupyter notebook demonstrating all DL components
- **File**: `notebooks/presentation_demo.ipynb`
- **Sections**:
  1. **Introduction**: Problem statement, mission
  2. **Dataset Overview**: FoodSeg103, Food-101, FoodX-251
  3. **Detection Model**: 
     - Architecture (YOLOv8)
     - Training process
     - Results (before/after improvement)
     - Visualizations
  4. **Classification Model**:
     - Architecture (EfficientNet)
     - Transfer learning approach
     - Results
     - Confusion matrix
  5. **Pipeline Integration**:
     - End-to-end flow
     - Example predictions
     - Performance metrics
  6. **Deep Learning Techniques Used**:
     - Object detection (YOLO)
     - Semantic segmentation
     - Transfer learning
     - Data augmentation
     - Multi-stage pipeline
  7. **Results & Analysis**:
     - Quantitative metrics
     - Qualitative examples
     - Failure cases
     - Future improvements

**Task 3.8: Prepare Demo Script** (1 hour)
- **Action**: Script to run live demo during presentation
- **File**: `demo.py`
- **Features**:
  - Load test images
  - Run pipeline
  - Display results with visualizations
  - Show metrics

**Day 3 Deliverables**:
- ✅ Complete end-to-end pipeline (Detection → Classification → Nutrition)
- ✅ Nutrition API integration
- ✅ Model analysis and visualizations
- ✅ Evaluation metrics
- ✅ Presentation notebook
- ✅ Demo script

---

## Deep Learning Techniques to Emphasize in Presentation

### 1. **Object Detection & Segmentation (YOLOv8)**
- **Architecture**: Single-stage detector with segmentation head
- **Key Features**: 
  - Anchor-free detection
  - Multi-scale feature pyramid
  - Instance segmentation
- **Why Important**: Detects multiple foods in single image, provides pixel-level masks

### 2. **Transfer Learning**
- **Pre-trained Models**: 
  - YOLOv8 pretrained on COCO
  - EfficientNet pretrained on ImageNet
- **Fine-tuning Strategy**: 
  - Freeze backbone, train head → Full fine-tuning
  - Learning rate scheduling
- **Why Important**: Leverages large-scale pretraining, faster convergence

### 3. **Data Augmentation**
- **Techniques Used**:
  - Geometric: rotation, scaling, flipping
  - Photometric: brightness, contrast, saturation
  - Advanced: mosaic, mixup
- **Why Important**: Improves generalization, handles real-world variations

### 4. **Multi-Stage Deep Learning Pipeline**
- **Stage 1**: Detection (YOLOv8) - finds food regions
- **Stage 2**: Classification (EfficientNet) - identifies food type
- **Stage 3**: Nutrition mapping (API) - gets nutritional data
- **Why Important**: Modular approach, each stage optimized for specific task

### 5. **CNN Architecture (EfficientNet)**
- **Compound Scaling**: Balances depth, width, resolution
- **MobileNet blocks**: Depthwise separable convolutions
- **Why Important**: Efficient architecture, good accuracy/speed trade-off

### 6. **Loss Functions & Optimization**
- **Detection**: Box loss, segmentation loss, classification loss
- **Classification**: Cross-entropy with label smoothing
- **Optimizer**: AdamW with cosine annealing
- **Why Important**: Proper loss design crucial for multi-task learning

### 7. **Model Interpretability**
- **Grad-CAM**: Visualize attention
- **Confusion Matrix**: Understand failure modes
- **Feature Visualization**: What model learned
- **Why Important**: Builds trust, helps debugging

---

## Success Metrics for Presentation

### Quantitative Metrics:
- **Detection**: mAP50 > 0.40 (2x improvement from baseline)
- **Classification**: Accuracy > 85% on Food-101
- **End-to-End**: Correct food identification > 70%

### Qualitative Metrics:
- Visualizations showing accurate predictions
- Clear improvement from baseline
- Demonstrates understanding of DL concepts

### Presentation Quality:
- Clear explanation of architectures
- Demonstrates deep learning techniques
- Shows quantitative improvements
- Discusses challenges and solutions

---

## Risk Mitigation (2-3 Day Timeline)

**Risk 1: Training takes too long**
- *Mitigation*: Start training early, use smaller models if needed, reduce epochs if time-constrained

**Risk 2: Models don't converge**
- *Mitigation*: Use pretrained weights, proper learning rate, check data quality

**Risk 3: Integration issues**
- *Mitigation*: Test each component independently, use simple interfaces

**Risk 4: Not enough time for all features**
- *Mitigation*: Prioritize core DL components (detection + classification), nutrition can be simplified

---

## Daily Checklist

### Day 1:
- [ ] Upgrade YOLOv8 model architecture
- [ ] Configure extended training (50-100 epochs)
- [ ] Start training job
- [ ] Create detection inference script
- [ ] Monitor training progress

### Day 2:
- [ ] Set up EfficientNet classification
- [ ] Train on Food-101
- [ ] Fine-tune on FoodX-251
- [ ] Integrate detection + classification
- [ ] Test end-to-end pipeline

### Day 3:
- [ ] Integrate nutrition API
- [ ] Create evaluation scripts
- [ ] Generate visualizations
- [ ] Prepare presentation notebook
- [ ] Test demo script

---

## Final Deliverables

1. **Improved Detection Model**: YOLOv8s/m trained 50-100 epochs
2. **Classification Model**: EfficientNet trained on Food-101, fine-tuned on FoodX-251
3. **Complete Pipeline**: Detection → Classification → Nutrition
4. **Evaluation Metrics**: Comprehensive performance analysis
5. **Visualizations**: Predictions, attention maps, training curves
6. **Presentation Notebook**: Complete demo of all DL components
7. **Code Repository**: Well-organized, documented code

**Focus**: Deep Learning techniques, model improvements, quantitative results

**Mobile App**: Can be simple demo (just calls API) - not the focus for DL class

