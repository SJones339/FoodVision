#!/usr/bin/env python3
"""
SIMPLE YOLOv8 Training - Just run this!
This will download and convert the dataset automatically.
"""
from ultralytics import YOLO
from pathlib import Path
import sys

def main():
    print("=" * 60)
    print("TRAINING YOLOV8 FOR BETTER BOUNDING BOXES")
    print("=" * 60)
    
    # Auto-detect device and recommend model size
    import torch
    if torch.cuda.is_available():
        device = 'cuda'
        recommended = 'm'  # Medium model for CUDA
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        recommended = 'm'  # Medium model for MPS (Apple Silicon)
    else:
        device = 'cpu'
        recommended = 's'  # Small model for CPU
    
    print(f"\nDetected device: {device}")
    print(f"Recommended model size: YOLOv8{recommended} (good balance)")
    print("\nModel sizes:")
    print("  'n' (nano)   - Fastest, lowest accuracy (~2-3 hours)")
    print("  's' (small)   - Fast, good accuracy (~3-4 hours) ← Good default")
    print("  'm' (medium)  - Slower, better accuracy (~5-7 hours) ← Recommended for GPU")
    print("  'l' (large)   - Slow, best accuracy (~8-12 hours)")
    
    model_size = input(f"\nEnter model size [n/s/m/l] (default: {recommended}): ").strip().lower()
    if not model_size or model_size not in ['n', 's', 'm', 'l']:
        model_size = recommended
    
    print(f"\n✓ Using YOLOv8{model_size}-seg")
    print("\nThis will:")
    print("1. Download FoodSeg103 dataset (if needed)")
    print("2. Convert to YOLO format")
    print("3. Train YOLOv8 on food data")
    print(f"4. Save trained model to models/yolov8{model_size}_food_best.pt")
    
    if model_size == 'm':
        print("\n⏱️  Estimated time: 2-3 hours on GPU (20 epochs)")
    elif model_size == 's':
        print("\n⏱️  Estimated time: 1.5-2 hours on GPU (20 epochs)")
    else:
        print("\n⏱️  Estimated time: varies")
    
    print("\nLet's start!\n")
    
    # Step 1: Check/convert dataset
    # Check both possible locations
    data_yaml = Path("data/foodseg_pp/data.yaml")
    alt_data_yaml = Path("datasets/foodseg_pp/data.yaml")
    
    if not data_yaml.exists() and not alt_data_yaml.exists():
        print("⚠️  Dataset not found. Converting FoodSeg103 to YOLO format...")
        print("   (This takes 10-30 minutes - be patient!)\n")
        
        # Import and run conversion
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from datasets import load_dataset
        from utils.download_datasets import convert_foodseg_to_yolo
        
        print("Loading FoodSeg103 from HuggingFace...")
        try:
            dataset = load_dataset("EduardoPacheco/FoodSeg103")
        except:
            # Try with trust_remote_code if needed
            dataset = load_dataset("EduardoPacheco/FoodSeg103", trust_remote_code=True)
        
        print("Converting to YOLO format...")
        convert_foodseg_to_yolo(dataset, Path("datasets/foodseg103"))
        
        # Check again after conversion
        if alt_data_yaml.exists():
            data_yaml = alt_data_yaml
        elif not data_yaml.exists():
            print("\n❌ Conversion failed. Check errors above.")
            return
    
    # Use whichever path exists
    if alt_data_yaml.exists() and not data_yaml.exists():
        data_yaml = alt_data_yaml
        print(f"✓ Found dataset at: {data_yaml}")
    elif data_yaml.exists():
        print(f"✓ Found dataset at: {data_yaml}")
    
    print("✓ Dataset ready!\n")
    
    # Step 2: Train YOLOv8
    model_name = f'yolov8{model_size}-seg.pt'
    print(f"Loading {model_name}...")
    model = YOLO(model_name)
    
    print(f"Using device: {device}\n")
    print("Starting training...")
    print("This will take 1.5-3 hours (20 epochs). Progress will be shown below.\n")
    
    # Adjust batch size based on model size (reduced for stability)
    if model_size == 'n':
        batch_size = 16  # Reduced from 32 for stability
    elif model_size == 's':
        batch_size = 8   # Reduced from 16 for stability
    elif model_size == 'm':
        batch_size = 6   # Reduced from 12 for stability
    else:  # 'l' or 'x'
        batch_size = 4   # Reduced from 8 for stability
    
    print(f"Batch size: {batch_size} (reduced for stability with segmentation)")
    
    # Train with robust settings to avoid crashes
    print("Using conservative augmentation to avoid shape mismatch errors...")
    results = model.train(
        data=str(data_yaml),
        epochs=20,
        imgsz=640,
        batch=batch_size,
        device=device,
        workers=0,  # Avoid multiprocessing issues on macOS
        patience=15,  # Early stopping
        save=True,
        project='runs/detect',
        name=f'yolov8{model_size}_food',
        # Better detection parameters
        conf=0.25,
        iou=0.45,
        # Reduced augmentation to avoid shape mismatch errors
        hsv_h=0.01,      # Reduced from 0.015
        hsv_s=0.5,       # Reduced from 0.7
        hsv_v=0.3,       # Reduced from 0.4
        degrees=5,       # Reduced from 10
        translate=0.05,  # Reduced from 0.1
        scale=0.3,       # Reduced from 0.5
        fliplr=0.5,
        mosaic=0.5,      # Reduced from 1.0 (mosaic can cause shape issues)
        mixup=0.0,       # DISABLED - causes shape mismatch with segmentation
        copy_paste=0.0,  # DISABLED - can cause issues
        erasing=0.0,     # DISABLED - random erasing
        # Additional stability settings
        amp=True,        # Mixed precision (more stable)
        close_mosaic=10, # Disable mosaic in last 10 epochs
    )
    
    # Copy best model to models directory
    best_model = Path(results.save_dir) / 'weights' / 'best.pt'
    output_path = Path(f'models/yolov8{model_size}_food_best.pt')
    output_path.parent.mkdir(exist_ok=True)
    
    import shutil
    shutil.copy(best_model, output_path)
    
    print("\n" + "=" * 60)
    print("✓ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nBest model saved to: {output_path}")
    print(f"Training results in: {results.save_dir}")
    print("\nTo use this model:")
    print(f"  pipeline = FoodVisionPipeline(")
    print(f"      detection_model_path='{output_path}'")
    print(f"  )")
    print("\nYour bounding boxes should now be MUCH better! 🎯")

if __name__ == "__main__":
    main()

