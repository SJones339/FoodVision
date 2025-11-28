#!/usr/bin/env python3
"""
Quick dataset download script for FoodVision project.

Run this from the project root:
    python download_datasets.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.download_datasets import (
    download_foodseg103,
    download_food101,
    download_foodx251
)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download FoodVision datasets")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--foodseg103", action="store_true", help="Download FoodSeg103")
    parser.add_argument("--food101", action="store_true", help="Download Food-101")
    parser.add_argument("--foodx251", action="store_true", help="Download FoodX-251")
    parser.add_argument("--convert-yolo", action="store_true", help="Convert FoodSeg103 to YOLO format")
    parser.add_argument("--roboflow-key", help="Roboflow API key (or set ROBOFLOW_API_KEY env var)")
    
    args = parser.parse_args()
    
    output_dir = Path("datasets")
    output_dir.mkdir(exist_ok=True)
    
    if args.roboflow_key:
        import os
        os.environ["ROBOFLOW_API_KEY"] = args.roboflow_key
    
    if args.all or (not any([args.foodseg103, args.food101, args.foodx251])):
        print("Downloading all datasets...\n")
        download_foodseg103(output_dir, convert_to_yolo=args.convert_yolo)
        download_food101(output_dir)
        download_foodx251(output_dir)
    else:
        if args.foodseg103:
            download_foodseg103(output_dir, convert_to_yolo=args.convert_yolo)
        if args.food101:
            download_food101(output_dir)
        if args.foodx251:
            download_foodx251(output_dir)
    
    print("\n✓ Dataset download complete!")

