"""
USDA Nutrition API Integration
Maps Food-101 classification results to USDA FoodData Central nutrition data
"""

import requests
import json
from typing import Dict, Optional, List
from pathlib import Path
import time


class NutritionAPI:
    """
    Connects Food-101 classification results to USDA nutrition data.
    
    Handles:
    - Food name normalization (Food-101 format -> USDA search terms)
    - USDA API queries
    - Nutrition data extraction
    - Caching for performance
    """
    
    def __init__(self, api_key: Optional[str] = None, cache_file: Optional[str] = None):
        """
        Initialize Nutrition API client.
        
        Args:
            api_key: USDA API key (if None, will try to load from env or config)
            cache_file: Path to JSON cache file for nutrition data
        """
        self.api_key = api_key or self._load_api_key()
        self.base_url = "https://api.nal.usda.gov/fdc/v1"
        self.cache_file = cache_file or "data/nutrition_cache.json"
        self.cache = self._load_cache()
        
        # Rate limiting (USDA API allows 1000 requests/hour)
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
        # Food-101 to USDA search term mappings
        self.food_mappings = self._build_food_mappings()
    
    def _load_api_key(self) -> Optional[str]:
        """Load API key from environment, .env file, or config file"""
        import os
        
        # First, try environment variable (already loaded)
        api_key = os.getenv('USDA_API_KEY')
        if api_key:
            return api_key
        
        # Try to load from .env file
        project_root = Path(__file__).parent.parent.parent
        env_file = project_root / ".env"
        if env_file.exists():
            try:
                with open(env_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('USDA_API_KEY='):
                            # Extract value after the = sign
                            key_value = line.split('=', 1)
                            if len(key_value) == 2:
                                api_key = key_value[1].strip().strip('"').strip("'")
                                if api_key:
                                    return api_key
            except Exception as e:
                print(f"⚠️  Error reading .env file: {e}")
        
        # Try to load from config file
        config_path = project_root / "config" / "usda_api_key.txt"
        if config_path.exists():
            try:
                return config_path.read_text().strip()
            except Exception as e:
                print(f"⚠️  Error reading config file: {e}")
        
        return None
    
    def _load_cache(self) -> Dict:
        """Load nutrition data cache from file"""
        cache_path = Path(self.cache_file)
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        """Save nutrition data cache to file"""
        cache_path = Path(self.cache_file)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(self.cache, f, indent=2)
    
    def _build_food_mappings(self) -> Dict[str, str]:
        """
        Build comprehensive mapping from Food-101 class names to USDA search terms.
        
        Food-101 uses underscores (e.g., 'apple_pie'), but USDA works better with
        natural language terms. This mapping handles:
        - Direct conversions (underscore -> space)
        - Special cases that need different search terms
        - Common food variations
        """
        mappings = {}
        
        # Get all Food-101 class names
        try:
            from datasets import load_dataset
            ds = load_dataset("ethz/food101", split='train')
            food101_classes = ds.features['label'].names
        except:
            # Fallback: use hardcoded list if dataset not available
            food101_classes = [
                'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
                'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
                'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake',
                'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla',
                'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder',
                'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes',
                'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict',
                'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras',
                'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice',
                'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich',
                'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup',
                'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna',
                'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup',
                'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters',
                'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck',
                'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib',
                'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto',
                'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits',
                'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake',
                'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles'
            ]
        
        # Build mappings: default is underscore -> space, but we can override
        special_mappings = {
            # Foods that need more specific USDA search terms
            'baby_back_ribs': 'pork ribs',
            'beef_carpaccio': 'beef raw',
            'beef_tartare': 'beef raw',
            'beet_salad': 'beets',
            'bread_pudding': 'bread pudding',
            'breakfast_burrito': 'burrito',
            'caesar_salad': 'caesar salad',
            'caprese_salad': 'caprese salad',
            'chicken_curry': 'chicken curry',
            'chicken_quesadilla': 'quesadilla',
            'chicken_wings': 'chicken wings',
            'chocolate_cake': 'chocolate cake',
            'chocolate_mousse': 'chocolate mousse',
            'clam_chowder': 'clam chowder',
            'club_sandwich': 'club sandwich',
            'crab_cakes': 'crab cakes',
            'creme_brulee': 'creme brulee',
            'croque_madame': 'croque madame',
            'cup_cakes': 'cupcakes',
            'deviled_eggs': 'deviled eggs',
            'fish_and_chips': 'fish and chips',
            'foie_gras': 'foie gras',
            'french_fries': 'french fries',
            'french_onion_soup': 'french onion soup',
            'french_toast': 'french toast',
            'fried_calamari': 'calamari',
            'fried_rice': 'fried rice',
            'frozen_yogurt': 'frozen yogurt',
            'garlic_bread': 'garlic bread',
            'greek_salad': 'greek salad',
            'grilled_cheese_sandwich': 'grilled cheese',
            'grilled_salmon': 'salmon grilled',
            'hot_and_sour_soup': 'hot and sour soup',
            'hot_dog': 'hot dog',
            'huevos_rancheros': 'huevos rancheros',
            'ice_cream': 'ice cream',
            'lobster_bisque': 'lobster bisque',
            'lobster_roll_sandwich': 'lobster roll',
            'macaroni_and_cheese': 'macaroni and cheese',
            'miso_soup': 'miso soup',
            'onion_rings': 'onion rings',
            'pad_thai': 'pad thai',
            'panna_cotta': 'panna cotta',
            'peking_duck': 'peking duck',
            'pork_chop': 'pork chop',
            'pulled_pork_sandwich': 'pulled pork',
            'red_velvet_cake': 'red velvet cake',
            'seaweed_salad': 'seaweed',
            'shrimp_and_grits': 'shrimp and grits',
            'spaghetti_bolognese': 'spaghetti bolognese',
            'spaghetti_carbonara': 'spaghetti carbonara',
            'spring_rolls': 'spring rolls',
            'strawberry_shortcake': 'strawberry shortcake',
            'tuna_tartare': 'tuna raw',
        }
        
        # Build complete mapping
        for food_class in food101_classes:
            if food_class in special_mappings:
                mappings[food_class] = special_mappings[food_class]
            else:
                # Default: replace underscores with spaces
                mappings[food_class] = food_class.replace('_', ' ')
        
        return mappings
    
    def normalize_food_name(self, food101_name: str) -> str:
        """
        Convert Food-101 class name to USDA search term.
        
        Args:
            food101_name: Food-101 class name (e.g., 'apple_pie', 'chicken_curry')
            
        Returns:
            USDA search term (e.g., 'apple pie', 'chicken curry')
        """
        food101_name = food101_name.lower().strip()
        
        # Check if we have a direct mapping
        if food101_name in self.food_mappings:
            return self.food_mappings[food101_name]
        
        # Fallback: replace underscores with spaces
        return food101_name.replace('_', ' ')
    
    def search_food(self, food_name: str, use_cache: bool = True) -> Optional[Dict]:
        """
        Search USDA database for food nutrition data.
        
        Args:
            food_name: Food name (Food-101 format or normalized)
            use_cache: Whether to use cached results
            
        Returns:
            Dict with nutrition data or None if not found
        """
        # Normalize food name
        normalized_name = self.normalize_food_name(food_name)
        
        # Check cache first
        cache_key = normalized_name.lower()
        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        # Check if API key is available
        if not self.api_key:
            print(f"⚠️  USDA API key not found. Skipping nutrition lookup for: {normalized_name}")
            print("   Set USDA_API_KEY environment variable or create config/usda_api_key.txt")
            return None
        
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        
        # Search USDA API
        url = f"{self.base_url}/foods/search"
        params = {
            'api_key': self.api_key,
            'query': normalized_name,
            'pageSize': 5,  # Get top 5 results
            'dataType': 'Foundation,SR Legacy'  # Prefer Foundation and SR Legacy data
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            self.last_request_time = time.time()
            
            if data.get('foods') and len(data['foods']) > 0:
                # Use the first result (most relevant)
                food = data['foods'][0]
                nutrition = self._extract_nutrition(food)
                
                # Cache the result
                self.cache[cache_key] = nutrition
                self._save_cache()
                
                return nutrition
            else:
                print(f"⚠️  No USDA results found for: {normalized_name}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Error fetching nutrition data for {normalized_name}: {e}")
            return None
        except Exception as e:
            print(f"⚠️  Unexpected error for {normalized_name}: {e}")
            return None
    
    def _extract_nutrition(self, food_data: Dict) -> Dict:
        """
        Extract key nutrition information from USDA API response.
        
        Args:
            food_data: Raw food data from USDA API
            
        Returns:
            Dict with normalized nutrition values
        """
        nutrients = {
            'fdc_id': food_data.get('fdcId'),
            'description': food_data.get('description', ''),
            'calories': None,
            'protein': None,
            'carbs': None,
            'fat': None,
            'fiber': None,
            'sugar': None,
            'sodium': None,
        }
        
        # Extract nutrients from foodNutrients array
        food_nutrients = food_data.get('foodNutrients', [])
        
        for nutrient in food_nutrients:
            nutrient_name = nutrient.get('nutrientName', '').lower()
            nutrient_value = nutrient.get('value')
            unit = nutrient.get('unitName', '').lower()
            
            if nutrient_value is None:
                continue
            
            # Map USDA nutrient names to our standard fields
            if 'energy' in nutrient_name or 'calories' in nutrient_name:
                if nutrients['calories'] is None:
                    nutrients['calories'] = nutrient_value
            elif 'protein' in nutrient_name and 'total' in nutrient_name:
                nutrients['protein'] = nutrient_value
            elif 'carbohydrate' in nutrient_name and 'total' in nutrient_name:
                nutrients['carbs'] = nutrient_value
            elif 'fat' in nutrient_name and 'total' in nutrient_name:
                nutrients['fat'] = nutrient_value
            elif 'fiber' in nutrient_name and 'total' in nutrient_name:
                nutrients['fiber'] = nutrient_value
            elif 'sugar' in nutrient_name and 'total' in nutrient_name:
                nutrients['sugar'] = nutrient_value
            elif 'sodium' in nutrient_name:
                nutrients['sodium'] = nutrient_value
        
        return nutrients
    
    def get_nutrition_for_items(self, items: List[Dict]) -> List[Dict]:
        """
        Get nutrition data for a list of food items.
        
        Args:
            items: List of items with 'food_name' field (from pipeline results)
            
        Returns:
            List of items with added nutrition data
        """
        results = []
        for item in items:
            food_name = item.get('food_name', '')
            nutrition = self.search_food(food_name)
            
            # Add nutrition data to item
            item_with_nutrition = item.copy()
            if nutrition:
                item_with_nutrition['nutrition'] = nutrition
                item_with_nutrition['calories'] = nutrition.get('calories')
                item_with_nutrition['protein'] = nutrition.get('protein')
                item_with_nutrition['carbs'] = nutrition.get('carbs')
                item_with_nutrition['fat'] = nutrition.get('fat')
            else:
                item_with_nutrition['nutrition'] = None
                item_with_nutrition['calories'] = None
            
            results.append(item_with_nutrition)
        
        return results


if __name__ == "__main__":
    # Test the nutrition API
    api = NutritionAPI()
    
    # Test with some Food-101 class names
    test_foods = ['apple_pie', 'chicken_curry', 'pizza', 'grilled_salmon']
    
    print("Testing Food-101 to USDA mapping:")
    print("=" * 60)
    for food in test_foods:
        normalized = api.normalize_food_name(food)
        print(f"{food:30} -> {normalized}")
    
    print("\n" + "=" * 60)
    print("Testing nutrition lookup (requires API key):")
    for food in test_foods[:2]:  # Test first 2 to avoid rate limits
        print(f"\nSearching for: {food}")
        nutrition = api.search_food(food)
        if nutrition:
            print(f"  Found: {nutrition.get('description', 'N/A')}")
            print(f"  Calories: {nutrition.get('calories', 'N/A')}")
            print(f"  Protein: {nutrition.get('protein', 'N/A')}g")
            print(f"  Carbs: {nutrition.get('carbs', 'N/A')}g")
            print(f"  Fat: {nutrition.get('fat', 'N/A')}g")
        else:
            print("  No nutrition data found")

