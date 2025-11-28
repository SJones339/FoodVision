"""
Dataset Download Script for FoodVision Project

This script downloads and prepares the three main datasets:
1. FoodSeg103 - Segmentation dataset from HuggingFace
2. Food-101 - Classification dataset from HuggingFace
3. FoodX-251 - Fine-grained classification dataset from Roboflow

Usage:
    python src/utils/download_datasets.py [--dataset DATASET_NAME] [--all]
    
    Options:
        --dataset: Download specific dataset (foodseg103, food101, foodx251)
        --all: Download all datasets
        --convert-foodseg: Convert FoodSeg103 to YOLO format after download
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Try to import required packages, install if missing
try:
    import yaml
except ImportError:
    print("Installing pyyaml...")
    os.system("pip install pyyaml")
    import yaml

try:
    import cv2
except ImportError:
    print("Installing opencv-python...")
    os.system("pip install opencv-python")
    import cv2

try:
    import numpy as np
except ImportError:
    print("Installing numpy...")
    os.system("pip install numpy")
    import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    print("Installing tqdm...")
    os.system("pip install tqdm")
    from tqdm import tqdm

try:
    from datasets import load_dataset
except ImportError:
    print("Installing datasets...")
    os.system("pip install datasets")
    from datasets import load_dataset

try:
    from roboflow import Roboflow
except ImportError:
    print("Installing roboflow...")
    os.system("pip install roboflow")
    from roboflow import Roboflow


def download_foodseg103(output_dir, convert_to_yolo=False):
    """Download FoodSeg103 dataset from HuggingFace."""
    print("\n=== Downloading FoodSeg103 Dataset ===")
    output_path = Path(output_dir) / "foodseg103"
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        print("Loading dataset from HuggingFace...")
        ds = load_dataset("EduardoPacheco/FoodSeg103", trust_remote_code=True)
        
        print(f"Saving to {output_path}...")
        ds.save_to_disk(str(output_path))
        print(f"✓ FoodSeg103 downloaded successfully to {output_path}")
        
        if convert_to_yolo:
            print("\nConverting FoodSeg103 to YOLO format...")
            convert_foodseg_to_yolo(ds, output_path)
        
        return output_path
    except Exception as e:
        print(f"✗ Error downloading FoodSeg103: {e}")
        return None


def convert_foodseg_to_yolo(dataset, output_dir):
    """Convert FoodSeg103 semantic masks to YOLO polygon format."""
    output_dir = Path(output_dir)
    yolo_dir = output_dir.parent / "foodseg_pp"
    
    train_img_dir = yolo_dir / "images" / "train"
    train_lbl_dir = yolo_dir / "labels" / "train"
    val_img_dir = yolo_dir / "images" / "val"
    val_lbl_dir = yolo_dir / "labels" / "val"
    
    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    def convert_split(split, img_dir, lbl_dir):
        print(f"Converting {len(split)} images...")
        for i, item in enumerate(tqdm(split, desc="Converting")):
            img = np.array(item["image"])
            mask = np.array(item["label"])
            H, W = mask.shape
            
            img_path = img_dir / f"{i}.jpg"
            cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
            label_path = lbl_dir / f"{i}.txt"
            with open(label_path, "w") as f:
                classes = np.unique(mask)
                classes = classes[classes != 0]  # remove background
                
                for cls in classes:
                    binmask = (mask == cls).astype(np.uint8)
                    contours, _ = cv2.findContours(
                        binmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    
                    for cnt in contours:
                        if len(cnt) < 3:
                            continue
                        
                        poly = cnt.reshape(-1, 2).astype(float)
                        poly[:, 0] /= W
                        poly[:, 1] /= H
                        poly = poly.flatten().tolist()
                        
                        line = f"{cls} " + " ".join([f"{p:.6f}" for p in poly]) + "\n"
                        f.write(line)
    
    print("Converting training split...")
    convert_split(dataset["train"], train_img_dir, train_lbl_dir)
    
    print("Converting validation split...")
    convert_split(dataset["validation"], val_img_dir, val_lbl_dir)
    
    # Create data.yaml
    yaml_path = yolo_dir / "data.yaml"
    data = {
        "path": str(yolo_dir.absolute()),
        "train": "images/train",
        "val": "images/val",
        "nc": 104,
        "names": [f"class_{i}" for i in range(104)]
    }
    
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, sort_keys=False)
    
    print(f"✓ YOLO format conversion complete. Saved to {yolo_dir}")
    print(f"✓ Created data.yaml at {yaml_path}")


def download_food101(output_dir):
    """Download Food-101 dataset from HuggingFace."""
    print("\n=== Downloading Food-101 Dataset ===")
    output_path = Path(output_dir) / "food101"
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        print("Loading dataset from HuggingFace...")
        ds = load_dataset("ethz/food101")
        
        print(f"Saving to {output_path}...")
        ds.save_to_disk(str(output_path))
        print(f"✓ Food-101 downloaded successfully to {output_path}")
        return output_path
    except Exception as e:
        print(f"✗ Error downloading Food-101: {e}")
        return None


def download_foodx251(output_dir, api_key=None):
    """Download FoodX-251 dataset from Roboflow."""
    print("\n=== Downloading FoodX-251 Dataset ===")
    output_path = Path(output_dir) / "foodx251"
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not api_key:
        api_key = os.getenv("ROBOFLOW_API_KEY")
        if not api_key:
            print("⚠ Roboflow API key not found!")
            print("Please set ROBOFLOW_API_KEY environment variable or pass --roboflow-key")
            print("You can get an API key from: https://roboflow.com/")
            return None
    
    try:
        print("Connecting to Roboflow...")
        rf = Roboflow(api_key=api_key)
        project = rf.workspace("foodx251").project("foodx-251")
        version = project.version(4)
        
        print(f"Downloading to {output_path}...")
        dataset = version.download(
            model_format="folder",
            location=str(output_path)
        )
        
        print(f"✓ FoodX-251 downloaded successfully to {output_path}")
        return output_path
    except Exception as e:
        print(f"✗ Error downloading FoodX-251: {e}")
        print("Make sure your Roboflow API key is valid.")
        return None


def main():
    parser = argparse.ArgumentParser(description="Download datasets for FoodVision project")
    parser.add_argument(
        "--dataset",
        choices=["foodseg103", "food101", "foodx251"],
        help="Download specific dataset"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all datasets"
    )
    parser.add_argument(
        "--output-dir",
        default="datasets",
        help="Output directory for datasets (default: datasets/)"
    )
    parser.add_argument(
        "--convert-foodseg",
        action="store_true",
        help="Convert FoodSeg103 to YOLO format after download"
    )
    parser.add_argument(
        "--roboflow-key",
        help="Roboflow API key (or set ROBOFLOW_API_KEY env var)"
    )
    
    args = parser.parse_args()
    
    # Set Roboflow key if provided
    if args.roboflow_key:
        os.environ["ROBOFLOW_API_KEY"] = args.roboflow_key
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.all:
        print("Downloading all datasets...\n")
        download_foodseg103(output_dir, convert_to_yolo=args.convert_foodseg)
        download_food101(output_dir)
        download_foodx251(output_dir)
    elif args.dataset:
        if args.dataset == "foodseg103":
            download_foodseg103(output_dir, convert_to_yolo=args.convert_foodseg)
        elif args.dataset == "food101":
            download_food101(output_dir)
        elif args.dataset == "foodx251":
            download_foodx251(output_dir)
    else:
        print("Please specify --dataset or --all")
        parser.print_help()


if __name__ == "__main__":
    main()

