"""
End-to-End Food Vision Pipeline
Combines YOLOv8 (detection) + EfficientNet (classification)

Pipeline:
1. YOLOv8 detects and segments food regions
2. Crop each detected region
3. EfficientNet classifies each region
4. Return results with food names and locations
"""

import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import numpy as np
from ultralytics import YOLO
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from efficientnet_pytorch import EfficientNet
except ImportError:
    print("Installing efficientnet-pytorch...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "efficientnet-pytorch"])
    from efficientnet_pytorch import EfficientNet

from torchvision import transforms


class FoodVisionPipeline:
    """
    Complete pipeline: Detection → Classification
    
    Usage:
        pipeline = FoodVisionPipeline(
            detection_model_path='path/to/yolo.pt',
            classification_model_path='models/efficientnet_best.pth'
        )
        results = pipeline.process_image('meal.jpg')
    """
    
    def __init__(self, 
                 detection_model_path=None,
                 classification_model_path='models/efficientnet_best.pth',
                 device='auto',
                 conf_threshold=0.25):
        """
        Initialize pipeline with detection and classification models.
        
        Args:
            detection_model_path: Path to YOLOv8 model (if None, uses pretrained)
            classification_model_path: Path to EfficientNet model
            device: 'auto', 'cuda', 'mps', or 'cpu'
            conf_threshold: Confidence threshold for detection
        """
        # Auto-detect device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Load detection model (YOLOv8)
        if detection_model_path and Path(detection_model_path).exists():
            print(f"Loading YOLOv8 from {detection_model_path}")
            self.detector = YOLO(detection_model_path)
        else:
            print("Using pretrained YOLOv8n-seg (for demo - use your trained model)")
            self.detector = YOLO('yolov8n-seg.pt')
        
        self.conf_threshold = conf_threshold
        
        # Load classification model (EfficientNet)
        print(f"Loading EfficientNet from {classification_model_path}")
        self.classifier = self._load_classifier(classification_model_path)
        
        # Image transforms for classification
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Load class names (Food-101)
        self.class_names = self._load_class_names()
        
        print("✓ Pipeline initialized!")
    
    def _load_classifier(self, model_path):
        """Load EfficientNet classifier"""
        model = EfficientNet.from_pretrained('efficientnet-b0')
        num_features = model._fc.in_features
        model._fc = nn.Linear(num_features, 101)  # 101 Food-101 classes
        
        # Load trained weights
        if Path(model_path).exists():
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            print(f"✓ Loaded trained weights from {model_path}")
        else:
            print(f"⚠️  Model file not found: {model_path}")
            print("   Using pretrained ImageNet weights (not fine-tuned)")
        
        model = model.to(self.device)
        model.eval()
        return model
    
    def _load_class_names(self):
        """Load Food-101 class names"""
        try:
            from datasets import load_dataset
            ds = load_dataset("ethz/food101", split='train')
            return ds.features['label'].names
        except:
            # Fallback: return generic names
            return [f"food_{i}" for i in range(101)]
    
    def detect_foods(self, image_path):
        """
        Step 1: Detect food regions using YOLOv8
        
        Returns:
            List of detections with bbox, mask, confidence
        """
        results = self.detector.predict(
            source=str(image_path),
            conf=self.conf_threshold,
            imgsz=640,
            save=False,
            verbose=False
        )
        
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            masks = results[0].masks if results[0].masks is not None else None
            
            for i in range(len(boxes)):
                detection = {
                    'bbox': boxes.xyxy[i].cpu().numpy().tolist(),  # [x1, y1, x2, y2]
                    'confidence': float(boxes.conf[i].cpu().numpy()),
                    'class_id': int(boxes.cls[i].cpu().numpy()),
                    'mask': masks.data[i].cpu().numpy() if masks else None
                }
                detections.append(detection)
        
        return detections
    
    def classify_food(self, image_crop):
        """
        Step 2: Classify food region using EfficientNet
        
        Args:
            image_crop: PIL Image of cropped food region
            
        Returns:
            Dict with 'class_name', 'confidence', 'top_k' predictions
        """
        # Convert to RGB if needed
        if image_crop.mode != 'RGB':
            image_crop = image_crop.convert('RGB')
        
        # Transform
        image_tensor = self.transform(image_crop).unsqueeze(0).to(self.device)
        
        # Classify
        with torch.no_grad():
            outputs = self.classifier(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, 5)  # Top 5
        
        # Get class names
        top_classes = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            top_classes.append({
                'class_name': self.class_names[idx.item()],
                'confidence': float(prob.item())
            })
        
        return {
            'predicted_class': top_classes[0]['class_name'],
            'confidence': top_classes[0]['confidence'],
            'top_k': top_classes
        }
    
    def process_image(self, image_path, save_crops=False):
        """
        Complete pipeline: Detect → Classify
        
        Args:
            image_path: Path to input image
            save_crops: Whether to save cropped regions for debugging
            
        Returns:
            Dict with detected foods and classifications
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load original image
        original_image = Image.open(image_path).convert('RGB')
        
        print(f"\nProcessing image: {image_path.name}")
        print("=" * 60)
        
        # Step 1: Detect food regions
        print("Step 1: Detecting food regions...")
        detections = self.detect_foods(image_path)
        print(f"  Found {len(detections)} food regions")
        
        if len(detections) == 0:
            print("  ⚠️  No food detected in image")
            return {
                'image_path': str(image_path),
                'num_detections': 0,
                'items': []
            }
        
        # Step 2: Classify each detected region
        print("\nStep 2: Classifying food items...")
        results = []
        
        for i, det in enumerate(detections):
            # Crop region from image
            bbox = det['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Ensure valid crop
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(original_image.width, x2)
            y2 = min(original_image.height, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            crop = original_image.crop((x1, y1, x2, y2))
            
            # Save crop if requested
            if save_crops:
                crop_path = Path('data/processed/crops') / f"crop_{i}_{image_path.stem}.jpg"
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                crop.save(crop_path)
            
            # Classify
            classification = self.classify_food(crop)
            
            result = {
                'item_id': i + 1,
                'bbox': bbox,
                'detection_confidence': det['confidence'],
                'food_name': classification['predicted_class'],
                'classification_confidence': classification['confidence'],
                'top_predictions': classification['top_k']
            }
            
            results.append(result)
            
            print(f"  Item {i+1}: {classification['predicted_class']} "
                  f"(det: {det['confidence']:.2f}, cls: {classification['confidence']:.2f})")
        
        print("\n" + "=" * 60)
        print(f"✓ Processed {len(results)} food items")
        
        return {
            'image_path': str(image_path),
            'num_detections': len(detections),
            'items': results
        }
    
    def visualize_results(self, image_path, results, save_path=None):
        """
        Visualize detection and classification results on image
        
        Args:
            image_path: Path to original image
            results: Results from process_image()
            save_path: Where to save visualization
        """
        try:
            from PIL import ImageDraw, ImageFont
        except ImportError:
            print("PIL not available for visualization")
            return
        
        image = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(image)
        
        # Try to load font
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        except:
            font = ImageFont.load_default()
        
        # Draw bounding boxes and labels
        for item in results['items']:
            bbox = item['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline='green', width=3)
            
            # Draw label
            label = f"{item['food_name']} ({item['classification_confidence']:.2f})"
            draw.rectangle([x1, y1-25, x1+len(label)*8, y1], fill='green', outline='green')
            draw.text((x1+5, y1-20), label, fill='white', font=font)
        
        if save_path:
            image.save(save_path)
            print(f"✓ Saved visualization to {save_path}")
        else:
            image.show()
        
        return image


if __name__ == "__main__":
    # Example usage
    pipeline = FoodVisionPipeline(
        classification_model_path='models/efficientnet_best.pth'
    )
    
    # Test on a sample image (if you have one)
    # results = pipeline.process_image('path/to/meal.jpg')
    # print(results)
    
    print("\nPipeline ready! Use it like:")
    print("  pipeline = FoodVisionPipeline()")
    print("  results = pipeline.process_image('meal.jpg')")

