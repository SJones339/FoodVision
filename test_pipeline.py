#!/usr/bin/env python3
"""
Test the end-to-end pipeline: YOLOv8 + EfficientNet
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline.end_to_end import FoodVisionPipeline
import json

def test_pipeline(image_path=None):
    """Test the complete pipeline"""
    
    print("=" * 60)
    print("FOOD VISION PIPELINE TEST")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = FoodVisionPipeline(
        classification_model_path='models/efficientnet_best.pth'
    )
    
    # Test image
    if image_path is None:
        # Try to find a test image
        test_images = [
            'test_image.jpg',
            'sample.jpg',
            'meal.jpg',
            'data/raw/test.jpg'
        ]
        
        image_path = None
        for path in test_images:
            if Path(path).exists():
                image_path = path
                break
        
        if image_path is None:
            print("\n⚠️  No test image found!")
            print("   Place a test image in the project root or specify with:")
            print("   python test_pipeline.py path/to/image.jpg")
            return
    
    # Process image
    results = pipeline.process_image(image_path, save_crops=True)
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(json.dumps(results, indent=2))
    
    # Visualize
    print("\nCreating visualization...")
    vis_path = Path('models/pipeline_result.jpg')
    pipeline.visualize_results(image_path, results, save_path=str(vis_path))
    
    print(f"\n✓ Results saved to models/pipeline_result.jpg")
    print(f"✓ Cropped regions saved to data/processed/crops/")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test end-to-end pipeline')
    parser.add_argument('image', nargs='?', help='Path to test image')
    args = parser.parse_args()
    
    test_pipeline(args.image)

