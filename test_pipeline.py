#!/usr/bin/env python3
"""
Test the end-to-end pipeline: YOLOv8 + EfficientNet
Now using YOUR TRAINED YOLOv8 model! 🎯
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline.end_to_end import FoodVisionPipeline
import json

def test_pipeline(image_path=None, use_trained_model=True, include_nutrition=False):
    """Test the complete pipeline with your trained YOLOv8 model"""
    
    print("=" * 70)
    print("FOOD VISION PIPELINE TEST")
    print("Using YOUR TRAINED YOLOv8 Model! 🚀")
    print("=" * 70)
    
    # Determine which detection model to use
    if use_trained_model:
        detection_model = 'models/yolov8m_food_best.pt'
        if not Path(detection_model).exists():
            print(f"\n⚠️  Trained model not found: {detection_model}")
            print("   Falling back to pretrained model...")
            detection_model = None
        else:
            print(f"\n✓ Using trained model: {detection_model}")
    else:
        detection_model = None
        print("\n⚠️  Using pretrained YOLOv8 (not your trained model)")
    
    # Initialize pipeline
    pipeline = FoodVisionPipeline(
        detection_model_path=detection_model,
        classification_model_path='models/efficientnet_best.pth',
        include_nutrition=include_nutrition
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
    print(f"\n{'='*70}")
    print(f"Processing: {Path(image_path).name}")
    print(f"{'='*70}\n")
    
    results = pipeline.process_image(image_path, save_crops=True, include_nutrition=include_nutrition)
    
    # Print detailed results
    print("\n" + "=" * 70)
    print("DETECTION & CLASSIFICATION RESULTS")
    print("=" * 70)
    
    if results['num_detections'] == 0:
        print("\n⚠️  No food items detected!")
        print("   Try:")
        print("   - A clearer image with visible food")
        print("   - Lower confidence threshold")
        return
    
    print(f"\nDetected {results['num_detections']} food item(s):\n")
    
    for i, item in enumerate(results['items'], 1):
        print(f"Item {i}: {item['food_name'].replace('_', ' ').title()}")
        print(f"  Detection Confidence: {item['detection_confidence']:.1%}")
        print(f"  Classification Confidence: {item['classification_confidence']:.1%}")
        print(f"  Bounding Box: {[int(x) for x in item['bbox']]}")
        
        # Show top predictions
        if len(item.get('top_predictions', [])) > 1:
            print(f"  Top 3 Predictions:")
            for pred in item['top_predictions'][:3]:
                print(f"    - {pred['class_name'].replace('_', ' ').title()}: {pred['confidence']:.1%}")
        
        # Show nutrition if available
        if item.get('calories'):
            print(f"  Nutrition:")
            print(f"    Calories: {item['calories']:.0f}")
            if item.get('protein'):
                print(f"    Protein: {item['protein']:.1f}g")
            if item.get('carbs'):
                print(f"    Carbs: {item['carbs']:.1f}g")
            if item.get('fat'):
                print(f"    Fat: {item['fat']:.1f}g")
        print()
    
    # Nutrition summary
    summary = results.get('nutrition_summary', {})
    if summary.get('total_calories'):
        print("=" * 70)
        print("NUTRITION SUMMARY")
        print("=" * 70)
        print(f"Total Calories: {summary['total_calories']:.0f}")
        if summary.get('total_protein'):
            print(f"Total Protein: {summary['total_protein']:.1f}g")
        if summary.get('total_carbs'):
            print(f"Total Carbs: {summary['total_carbs']:.1f}g")
        if summary.get('total_fat'):
            print(f"Total Fat: {summary['total_fat']:.1f}g")
        print()
    
    # Visualize
    print("=" * 70)
    print("Creating visualization...")
    vis_path = Path('models/pipeline_result.jpg')
    pipeline.visualize_results(image_path, results, save_path=str(vis_path))
    
    print(f"\n{'='*70}")
    print("✓ TEST COMPLETE!")
    print(f"{'='*70}")
    print(f"✓ Visualization saved to: {vis_path}")
    print(f"✓ Cropped regions saved to: data/processed/crops/")
    print(f"✓ Full JSON results available in results variable")
    
    # Save JSON results
    json_path = Path('models/pipeline_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ JSON results saved to: {json_path}")
    
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Test end-to-end pipeline with your trained YOLOv8 model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with default image (meal.jpg)
  python test_pipeline.py
  
  # Test with specific image
  python test_pipeline.py my_meal.jpg
  
  # Test with nutrition data (requires USDA API key)
  python test_pipeline.py meal.jpg --nutrition
  
  # Use pretrained model instead of trained model
  python test_pipeline.py meal.jpg --pretrained
        """
    )
    parser.add_argument('image', nargs='?', help='Path to test image (default: meal.jpg)')
    parser.add_argument('--pretrained', action='store_true', 
                       help='Use pretrained YOLOv8 instead of trained model')
    parser.add_argument('--nutrition', action='store_true',
                       help='Include nutrition data lookup (requires USDA API key)')
    args = parser.parse_args()
    
    test_pipeline(
        image_path=args.image,
        use_trained_model=not args.pretrained,
        include_nutrition=args.nutrition
    )

