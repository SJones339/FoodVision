#!/usr/bin/env python3
"""
Convert existing FoodSeg103 dataset to YOLO format.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.download_datasets import convert_foodseg_to_yolo

if __name__ == "__main__":
    dataset_path = Path("datasets/foodseg103")
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found at {dataset_path}")
        print("Please download it first: python download_datasets.py --foodseg103")
        sys.exit(1)
    
    print(f"Loading dataset from HuggingFace (will use cached version if available)...")
    from datasets import load_dataset
    dataset = load_dataset("EduardoPacheco/FoodSeg103", trust_remote_code=True)
    
    print("Converting to YOLO format...")
    convert_foodseg_to_yolo(dataset, dataset_path)
    
    print("\n✓ Conversion complete!")
    print("You can now train YOLOv8 with:")
    print("  python src/detection/train_yolo.py --device mps")

