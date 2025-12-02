#!/usr/bin/env python3
"""
Simple script to train Mask R-CNN for food detection.

Mask R-CNN typically achieves 50-70% mAP (vs YOLOv8's 34%).
This is a better alternative for higher accuracy.

Usage:
    python train_mask_rcnn.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from detection.train_mask_rcnn import train_mask_rcnn


def main():
    print("=" * 70)
    print("MASK R-CNN TRAINING FOR FOOD DETECTION")
    print("Expected: 50-70% mAP (vs YOLOv8's 34%)")
    print("=" * 70)
    print()
    
    # Check if dataset exists
    data_yaml = Path('datasets/foodseg_pp/data.yaml')
    if not data_yaml.exists():
        print("⚠️  Dataset not found!")
        print("   Please download and convert FoodSeg103 first:")
        print("   python download_datasets.py --foodseg103 --convert-yolo")
        return
    
    # Train
    # Note: Mask R-CNN is often SLOWER on MPS than CPU!
    # Try 'cpu' if MPS is too slow
    print("\n⚠️  IMPORTANT: Mask R-CNN can be very slow on MPS (Apple GPU)")
    print("   If training is stuck, try: device='cpu' instead")
    print("   CPU is often faster for Mask R-CNN on Apple Silicon\n")
    
    train_mask_rcnn(
        data_yaml=str(data_yaml),
        num_classes=105,  # 104 food classes + background
        epochs=30,  # Reduced from 50 - can increase if needed
        batch_size=2,  # Reduced to 2 for faster training (was 4)
        lr=0.001,
        device='cpu'  # Use CPU - often faster than MPS for Mask R-CNN!
    )
    
    print("\n" + "=" * 70)
    print("✓ TRAINING COMPLETE!")
    print("=" * 70)
    print("\nTo use the trained model:")
    print("  from src.detection.mask_rcnn_inference import MaskRCNNFoodDetector")
    print("  detector = MaskRCNNFoodDetector('models/mask_rcnn_food_best.pth')")
    print("  results = detector.detect('meal.jpg')")


if __name__ == "__main__":
    main()

