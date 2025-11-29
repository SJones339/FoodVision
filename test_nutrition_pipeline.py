"""
Test script for Food-101 to USDA nutrition mapping
Demonstrates the complete pipeline with nutrition data integration
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from nutrition_pipeline.nutrition_api import NutritionAPI

# Try to import pipeline (optional, may not be available)
try:
    from pipeline.end_to_end import FoodVisionPipeline
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    FoodVisionPipeline = None


def test_food_mapping():
    """Test Food-101 to USDA name mapping"""
    print("=" * 70)
    print("Testing Food-101 to USDA Name Mapping")
    print("=" * 70)
    
    api = NutritionAPI()
    
    # Test various Food-101 class names
    test_foods = [
        'apple_pie',
        'chicken_curry',
        'pizza',
        'grilled_salmon',
        'macaroni_and_cheese',
        'french_fries',
        'beef_tartare',
        'baby_back_ribs',
        'spaghetti_carbonara',
        'ice_cream'
    ]
    
    print(f"\nTotal mappings available: {len(api.food_mappings)}")
    print("\nSample mappings:")
    print("-" * 70)
    for food in test_foods:
        normalized = api.normalize_food_name(food)
        print(f"  {food:35} -> {normalized}")
    
    print("\n✓ Mapping test complete!\n")


def test_nutrition_lookup():
    """Test USDA nutrition lookup (requires API key)"""
    print("=" * 70)
    print("Testing USDA Nutrition Lookup")
    print("=" * 70)
    
    api = NutritionAPI()
    
    if not api.api_key:
        print("\n⚠️  USDA API key not found.")
        print("   To test nutrition lookup:")
        print("   1. Get a free API key from: https://fdc.nal.usda.gov/api-guide.html")
        print("   2. Set environment variable: export USDA_API_KEY='your_key'")
        print("   3. Or create: config/usda_api_key.txt")
        return
    
    # Test with a few common foods
    test_foods = ['apple_pie', 'pizza', 'chicken_curry']
    
    print("\nTesting nutrition lookup:")
    print("-" * 70)
    for food in test_foods:
        print(f"\nSearching for: {food}")
        nutrition = api.search_food(food)
        
        if nutrition:
            print(f"  ✓ Found: {nutrition.get('description', 'N/A')}")
            print(f"    Calories: {nutrition.get('calories', 'N/A')}")
            print(f"    Protein: {nutrition.get('protein', 'N/A')}g")
            print(f"    Carbs: {nutrition.get('carbs', 'N/A')}g")
            print(f"    Fat: {nutrition.get('fat', 'N/A')}g")
        else:
            print(f"  ✗ No nutrition data found")
    
    print("\n✓ Nutrition lookup test complete!\n")


def test_complete_pipeline():
    """Test complete pipeline with nutrition integration"""
    print("=" * 70)
    print("Testing Complete Pipeline with Nutrition")
    print("=" * 70)
    
    if not PIPELINE_AVAILABLE:
        print("\n⚠️  Pipeline not available (missing dependencies)")
        print("   Skipping complete pipeline test")
        return
    
    # Check if we have a test image
    test_image = project_root / "meal.jpg"
    if not test_image.exists():
        print(f"\n⚠️  Test image not found: {test_image}")
        print("   Skipping complete pipeline test")
        return
    
    print(f"\nUsing test image: {test_image}")
    
    # Initialize pipeline with nutrition
    try:
        pipeline = FoodVisionPipeline(
            classification_model_path='models/efficientnet_best.pth',
            include_nutrition=True
        )
        
        # Process image
        results = pipeline.process_image(test_image, include_nutrition=True)
        
        # Display results
        print("\n" + "=" * 70)
        print("Pipeline Results:")
        print("=" * 70)
        print(f"Detected {results['num_detections']} food items\n")
        
        for item in results['items']:
            print(f"Item {item['item_id']}: {item['food_name']}")
            print(f"  Detection confidence: {item['detection_confidence']:.2f}")
            print(f"  Classification confidence: {item['classification_confidence']:.2f}")
            
            if item.get('calories'):
                print(f"  Nutrition:")
                print(f"    Calories: {item['calories']:.0f}")
                if item.get('protein'):
                    print(f"    Protein: {item['protein']:.1f}g")
                if item.get('carbs'):
                    print(f"    Carbs: {item['carbs']:.1f}g")
                if item.get('fat'):
                    print(f"    Fat: {item['fat']:.1f}g")
            else:
                print(f"  Nutrition: Not available")
            print()
        
        # Summary
        summary = results.get('nutrition_summary', {})
        if summary.get('total_calories'):
            print("=" * 70)
            print("Nutrition Summary:")
            print(f"  Total Calories: {summary['total_calories']:.0f}")
            if summary.get('total_protein'):
                print(f"  Total Protein: {summary['total_protein']:.1f}g")
            if summary.get('total_carbs'):
                print(f"  Total Carbs: {summary['total_carbs']:.1f}g")
            if summary.get('total_fat'):
                print(f"  Total Fat: {summary['total_fat']:.1f}g")
        
        print("\n✓ Complete pipeline test finished!\n")
        
    except Exception as e:
        print(f"\n⚠️  Error running pipeline: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("FoodVision Nutrition Pipeline Test")
    print("=" * 70)
    
    # Test 1: Food name mapping
    test_food_mapping()
    
    # Test 2: Nutrition lookup (requires API key)
    test_nutrition_lookup()
    
    # Test 3: Complete pipeline
    test_complete_pipeline()
    
    print("=" * 70)
    print("All tests complete!")
    print("=" * 70)

