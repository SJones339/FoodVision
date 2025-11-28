"""
Train YOLOv8 on FoodSeg103 for food detection and segmentation.

This script trains YOLOv8 to detect WHERE food is in images.
YOLOv8 can also classify, but EfficientNet does BETTER classification.

Architecture:
- YOLOv8: Detects food regions (bounding boxes + segmentation masks)
- EfficientNet: Fine-grained classification of detected regions

Why both?
- YOLOv8: Good at detection, weaker at fine-grained classification
- EfficientNet: Excellent at fine-grained classification (85% accuracy)
- Together: Best detection + best classification
"""

from ultralytics import YOLO
from pathlib import Path
import argparse
import yaml


def train_yolo_detection(
    model_size='s',  # 'n', 's', 'm', 'l', 'x'
    epochs=50,
    batch_size=16,
    img_size=640,
    data_yaml='data/foodseg_pp/data.yaml',
    output_dir='models/',
    device='auto'
):
    """
    Train YOLOv8 segmentation model on FoodSeg103.
    
    This model learns to:
    - Detect WHERE food is (bounding boxes)
    - Segment food regions (pixel-level masks)
    - Roughly classify food types (103 FoodSeg103 classes)
    
    But for fine-grained classification, we use EfficientNet (101 Food-101 classes).
    
    Args:
        model_size: YOLOv8 size ('n'=nano, 's'=small, 'm'=medium, etc.)
        epochs: Number of training epochs
        batch_size: Batch size
        img_size: Image size for training
        data_yaml: Path to dataset YAML file
        output_dir: Where to save the model
        device: 'auto', 'cuda', 'mps', or 'cpu'
    """
    print("=" * 60)
    print("TRAINING YOLOV8 FOR FOOD DETECTION")
    print("=" * 60)
    print(f"\nModel: YOLOv8{model_size}-seg")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Image Size: {img_size}")
    print(f"Dataset: {data_yaml}")
    
    # Check if data YAML exists
    if not Path(data_yaml).exists():
        print(f"\n⚠️  ERROR: Dataset YAML not found: {data_yaml}")
        print("   Make sure you've converted FoodSeg103 to YOLO format:")
        print("   python download_datasets.py --foodseg103 --convert-yolo")
        return None
    
    # Auto-detect device
    import torch
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    print(f"Device: {device}")
    
    # Load model
    model_name = f"yolov8{model_size}-seg.pt"
    print(f"\nLoading {model_name}...")
    model = YOLO(model_name)
    
    # Train
    print("\nStarting training...")
    print("This will take several hours. Progress will be shown below.\n")
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        workers=0,  # 0 to avoid multiprocessing issues on macOS
        patience=15,  # Early stopping
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        # Optimization
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        # Loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,
    )
    
    # Get best model path
    best_model_path = Path(results.save_dir) / 'weights' / 'best.pt'
    
    # Copy to models directory
    output_path = Path(output_dir) / f'yolov8{model_size}_food_best.pt'
    output_path.parent.mkdir(exist_ok=True)
    
    import shutil
    shutil.copy(best_model_path, output_path)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Best model saved to: {output_path}")
    print(f"Training results in: {results.save_dir}")
    print("\nTo use this model in the pipeline:")
    print(f"  pipeline = FoodVisionPipeline(")
    print(f"      detection_model_path='{output_path}'")
    print(f"  )")
    
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train YOLOv8 on FoodSeg103')
    parser.add_argument('--model-size', choices=['n', 's', 'm', 'l', 'x'], default='s',
                       help='YOLOv8 model size (n=nano, s=small, m=medium, etc.)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Image size for training')
    parser.add_argument('--data-yaml', default='data/foodseg_pp/data.yaml',
                       help='Path to dataset YAML file')
    parser.add_argument('--device', default='auto',
                       help='Device (auto/cuda/mps/cpu)')
    
    args = parser.parse_args()
    
    train_yolo_detection(
        model_size=args.model_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        data_yaml=args.data_yaml,
        device=args.device
    )

