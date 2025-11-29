#!/usr/bin/env python3
"""Quick status check for YOLO conversion"""
from pathlib import Path

yaml_path = Path("data/foodseg_pp/data.yaml")
train_dir = Path("data/foodseg_pp/images/train")
val_dir = Path("data/foodseg_pp/images/val")

if yaml_path.exists():
    print("✓ data.yaml exists!")
    print(f"  Location: {yaml_path}")
    
    # Count images
    train_count = len(list(train_dir.glob("*.jpg"))) if train_dir.exists() else 0
    val_count = len(list(val_dir.glob("*.jpg"))) if val_dir.exists() else 0
    
    print(f"  Training images: {train_count}")
    print(f"  Validation images: {val_count}")
    
    if train_count > 0 and val_count > 0:
        print("\n✓ Dataset is ready! You can now train YOLOv8:")
        print("  python src/detection/train_yolo.py --model-size s --epochs 50 --device mps")
    else:
        print("\n⚠️  Dataset structure exists but images are still being converted...")
else:
    print("❌ Conversion not complete yet.")
    print("   Run: python convert_foodseg_to_yolo.py")
    print("   (This takes 10-30 minutes)")

