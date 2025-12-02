"""
Mask R-CNN inference for food detection.

This module provides inference using trained Mask R-CNN model,
which typically achieves 50-70% mAP (vs YOLOv8's 34%).
"""

import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from PIL import Image
import numpy as np
from pathlib import Path


def get_model_instance_segmentation(num_classes):
    """Create Mask R-CNN model architecture"""
    model = maskrcnn_resnet50_fpn(weights=None)  # No pretrained weights
    
    # Replace classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Replace mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )
    
    return model


class MaskRCNNFoodDetector:
    """Mask R-CNN based food detector"""
    
    def __init__(self, model_path, num_classes=105, device='auto', conf_threshold=0.5):
        """
        Initialize Mask R-CNN detector.
        
        Args:
            model_path: Path to trained model weights
            num_classes: Number of classes (104 food + 1 background)
            device: 'auto', 'cuda', 'mps', or 'cpu'
            conf_threshold: Confidence threshold for detections
        """
        # Auto-detect device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load model
        print(f"Loading Mask R-CNN from {model_path}")
        self.model = get_model_instance_segmentation(num_classes)
        
        # Load trained weights
        if Path(model_path).exists():
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print("✓ Loaded trained weights")
        else:
            print(f"⚠️  Model file not found: {model_path}")
            print("   Using untrained model")
        
        self.model.to(self.device)
        self.model.eval()
        
        self.conf_threshold = conf_threshold
    
    def detect(self, image_path):
        """
        Detect food items in image.
        
        Args:
            image_path: Path to image
            
        Returns:
            List of detections with bbox, mask, confidence, class_id
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # Extract detections
        detections = []
        pred = predictions[0]
        
        boxes = pred['boxes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        masks = pred['masks'].cpu().numpy()
        
        for i in range(len(boxes)):
            if scores[i] < self.conf_threshold:
                continue
            
            # Convert mask from (1, H, W) to (H, W)
            mask = masks[i][0]
            mask = (mask > 0.5).astype(np.uint8)  # Threshold mask
            
            detection = {
                'bbox': boxes[i].tolist(),  # [x1, y1, x2, y2]
                'confidence': float(scores[i]),
                'class_id': int(labels[i] - 1),  # Convert back to 0-indexed
                'mask': mask
            }
            detections.append(detection)
        
        return detections


if __name__ == "__main__":
    # Example usage
    detector = MaskRCNNFoodDetector(
        model_path='models/mask_rcnn_food_best.pth',
        conf_threshold=0.5
    )
    
    results = detector.detect('meal.jpg')
    print(f"Detected {len(results)} food items")


