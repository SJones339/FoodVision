"""
YOLOv8-seg → (optional) MiDaS depth → USDA macros pipeline.

This is the simplified "final project" runtime path:
1) User uploads image
2) YOLOv8-seg segments + classifies each segment (FoodSeg103-style classes)
3) MiDaS estimates relative depth; we use it to estimate a rough volume per mask
4) USDA FoodData Central provides nutrition per 100g; we scale using grams estimated
   from volume, with a serving-size fallback when depth-derived grams looks too small.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from ultralytics import YOLO


@dataclass(frozen=True)
class FoodItemResult:
    item_id: int
    food_name: str
    confidence: float
    bbox_xyxy: List[float]
    volume_ml: Optional[float]
    grams_estimated: Optional[float]
    macros: Optional[Dict[str, float]]  # keys: cal, protein, carbs, fat
    usda_description: Optional[str] = None


def _default_foodseg103_names() -> Dict[int, str]:
    # From the notebook (inline mapping, no runtime downloads).
    return {
        0: "background",
        1: "candy",
        2: "egg tart",
        3: "french fries",
        4: "chocolate",
        5: "biscuit",
        6: "popcorn",
        7: "pudding",
        8: "ice cream",
        9: "cheese butter",
        10: "cake",
        11: "wine",
        12: "milkshake",
        13: "coffee",
        14: "juice",
        15: "milk",
        16: "tea",
        17: "almond",
        18: "red beans",
        19: "cashew",
        20: "dried cranberries",
        21: "soy",
        22: "walnut",
        23: "peanut",
        24: "egg",
        25: "apple",
        26: "date",
        27: "apricot",
        28: "avocado",
        29: "banana",
        30: "strawberry",
        31: "cherry",
        32: "blueberry",
        33: "raspberry",
        34: "mango",
        35: "olives",
        36: "peach",
        37: "lemon",
        38: "pear",
        39: "fig",
        40: "pineapple",
        41: "grape",
        42: "kiwi",
        43: "melon",
        44: "orange",
        45: "watermelon",
        46: "steak",
        47: "pork",
        48: "chicken duck",
        49: "sausage",
        50: "fried meat",
        51: "lamb",
        52: "sauce",
        53: "crab",
        54: "fish",
        55: "shellfish",
        56: "shrimp",
        57: "soup",
        58: "bread",
        59: "corn",
        60: "hamburg",
        61: "pizza",
        62: "hanamaki baozi",
        63: "wonton dumplings",
        64: "pasta",
        65: "noodles",
        66: "rice",
        67: "pie",
        68: "tofu",
        69: "eggplant",
        70: "potato",
        71: "garlic",
        72: "cauliflower",
        73: "tomato",
        74: "kelp",
        75: "seaweed",
        76: "spring onion",
        77: "rape",
        78: "ginger",
        79: "okra",
        80: "lettuce",
        81: "pumpkin",
        82: "cucumber",
        83: "white radish",
        84: "carrot",
        85: "asparagus",
        86: "bamboo shoots",
        87: "broccoli",
        88: "celery stick",
        89: "cilantro mint",
        90: "snow peas",
        91: "cabbage",
        92: "bean sprouts",
        93: "onion",
        94: "pepper",
        95: "green beans",
        96: "French beans",
        97: "king oyster mushroom",
        98: "shiitake",
        99: "enoki mushroom",
        100: "oyster mushroom",
        101: "white button mushroom",
        102: "salad",
        103: "other ingredients",
    }


def estimate_volume_ml(
    depth: np.ndarray,
    mask: np.ndarray,
    *,
    plate_mm: float = 250.0,
    plate_px: float = 600.0,
    min_pixels: int = 50,
) -> float:
    """
    Rough volume estimate from relative depth.
    Matches the notebook logic (baseline = 95th percentile depth in the mask).
    """
    ys, xs = np.where(mask)
    if len(xs) < min_pixels:
        return 0.0

    d = depth[ys, xs]
    baseline = np.percentile(d, 95)
    height = np.clip(baseline - d, 0, None)

    mm_per_px = plate_mm / plate_px
    volume_mm3 = float(np.sum(height) * (mm_per_px**2))
    return volume_mm3 / 1000.0  # ml


class MidasDepthEstimator:
    """
    Loads MiDaS via torch.hub (DPT_Large by default).
    Note: first run downloads weights.
    """

    def __init__(self, device: str = "auto", model_name: str = "DPT_Large"):
        import torch

        self.torch = torch
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # MiDaS DPT models require `timm` (and download weights on first run).
        # If `timm` isn't installed, raise a clear error so callers can fall back.
        try:
            self.model = torch.hub.load("intel-isl/MiDaS", model_name).to(self.device)
        except ModuleNotFoundError as e:
            if "timm" in str(e):
                raise ModuleNotFoundError(
                    "MiDaS requires the `timm` package. Install it with: pip install timm"
                ) from e
            raise
        self.model.eval()
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

    def predict_depth(self, image_rgb: np.ndarray) -> np.ndarray:
        import cv2

        H, W = image_rgb.shape[:2]
        with self.torch.no_grad():
            depth = self.model(self.transform(image_rgb).to(self.device)).squeeze().cpu().numpy()
        return cv2.resize(depth, (W, H))


class USDAMacrosClient:
    """
    USDA FoodData Central lookup + portion scaling (from notebook).
    """

    def __init__(self, api_key: str):
        self.api_key = api_key

    def get_macros_for_portion(
        self,
        *,
        food_name: str,
        volume_ml: Optional[float],
        density_g_per_ml: float = 1.0,
        serving_fallback_ratio: float = 0.5,
        timeout_s: int = 10,
    ) -> Tuple[Optional[Dict[str, float]], Optional[float], Optional[str]]:
        import requests

        r = requests.post(
            "https://api.nal.usda.gov/fdc/v1/foods/search",
            params={"api_key": self.api_key},
            json={"query": food_name, "pageSize": 5},
            timeout=timeout_s,
        )
        r.raise_for_status()
        data = r.json()
        foods = data.get("foods", [])
        if not foods:
            return None, None, None

        food = foods[0]

        grams_depth = None
        if volume_ml is not None:
            grams_depth = float(volume_ml) * float(density_g_per_ml)

        grams_serving = None
        serving_size = food.get("servingSize")
        serving_unit = (food.get("servingSizeUnit") or "").lower()
        if serving_size is not None:
            if serving_unit in {"g", "gram", "grams"}:
                grams_serving = float(serving_size)
            elif serving_unit in {"ml", "milliliter", "milliliters"}:
                grams_serving = float(serving_size)  # assume 1g/ml

        # choose grams estimate (notebook heuristic)
        grams = grams_depth
        if grams_serving is not None and grams_depth is not None and grams_depth < serving_fallback_ratio * grams_serving:
            grams = grams_serving
        elif grams is None:
            grams = grams_serving

        if grams is None:
            return None, None, food.get("description")

        macros_100g: Dict[str, float] = {}
        for n in food.get("foodNutrients", []):
            name = (n.get("nutrientName") or "").lower()
            unit = (n.get("unitName") or "").lower()
            val = n.get("value")
            if val is None:
                continue
            if "energy" in name and unit == "kcal":
                macros_100g["cal"] = float(val)
            elif "protein" in name:
                macros_100g["protein"] = float(val)
            elif "carbohydrate" in name:
                macros_100g["carbs"] = float(val)
            elif "total lipid" in name or ("fat" in name and "saturated" not in name):
                macros_100g["fat"] = float(val)

        if not macros_100g:
            return None, float(grams), food.get("description")

        scale = float(grams) / 100.0
        macros = {k: float(v) * scale for k, v in macros_100g.items()}
        return macros, float(grams), food.get("description")


class FoodVisionWebPipeline:
    """
    Production-ish pipeline for the simple webapp.
    """

    def __init__(
        self,
        *,
        yolo_model_path: Union[str, Path] = "models/yolov8m_food_best.pt",
        usda_api_key: Optional[str] = None,
        conf_threshold: float = 0.25,
        device: str = "auto",
        enable_depth: bool = True,
    ):
        self.yolo = YOLO(str(yolo_model_path))
        self.conf_threshold = conf_threshold
        self.food_names = _default_foodseg103_names()

        self.usda = USDAMacrosClient(usda_api_key) if usda_api_key else None
        if enable_depth:
            try:
                self.depth_estimator = MidasDepthEstimator(device=device)
            except ModuleNotFoundError as e:
                # Allow the app to run even if depth deps are missing.
                self.depth_estimator = None
                self.depth_init_error = str(e)
            except Exception as e:
                self.depth_estimator = None
                self.depth_init_error = f"Depth disabled (MiDaS init failed): {e}"
        else:
            self.depth_estimator = None
            self.depth_init_error = None

    def process_image(
        self,
        image: Union[str, Path, Image.Image],
    ) -> Dict[str, Any]:
        # Load image
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            img_pil = Image.open(image_path).convert("RGB")
            image_id = str(image_path)
        else:
            img_pil = image.convert("RGB")
            image_id = "<in-memory>"

        img_np = np.array(img_pil)
        H, W = img_np.shape[:2]

        # YOLO inference
        res = self.yolo(img_np, verbose=False, conf=self.conf_threshold)[0]
        boxes = res.boxes
        masks = res.masks

        if boxes is None or len(boxes) == 0:
            return {
                "image": image_id,
                "num_items": 0,
                "items": [],
                "totals": None,
            }

        classes = boxes.cls.cpu().numpy().astype(int).tolist()
        confs = boxes.conf.cpu().numpy().astype(float).tolist()
        bboxes = boxes.xyxy.cpu().numpy().astype(float).tolist()

        masks_np: List[np.ndarray] = []
        if masks is not None and masks.data is not None:
            # masks.data is (N, Hmask, Wmask) in tensor form; convert to bool at full image size if needed
            m = masks.data.cpu().numpy()
            # Ultralytics usually already matches original shape; if not, we still handle via resize using PIL
            for i in range(m.shape[0]):
                mask = m[i]
                if mask.shape[0] != H or mask.shape[1] != W:
                    mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize((W, H))
                    mask = np.array(mask_img).astype(np.float32) / 255.0
                masks_np.append(mask > 0.5)
        else:
            masks_np = [np.zeros((H, W), dtype=bool) for _ in range(len(classes))]

        # Depth (optional)
        depth = self.depth_estimator.predict_depth(img_np) if self.depth_estimator else None

        items: List[FoodItemResult] = []
        totals = {"cal": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0}
        totals_any = False

        for i, (cls_id, conf, bbox) in enumerate(zip(classes, confs, bboxes), start=1):
            food_name = self.food_names.get(cls_id, f"class_{cls_id}")
            mask = masks_np[i - 1] if i - 1 < len(masks_np) else np.zeros((H, W), dtype=bool)

            volume_ml: Optional[float] = None
            if depth is not None:
                volume_ml = float(estimate_volume_ml(depth, mask))

            macros = None
            grams = None
            usda_desc = None
            if self.usda is not None:
                macros, grams, usda_desc = self.usda.get_macros_for_portion(
                    food_name=food_name,
                    volume_ml=volume_ml,
                )

            if macros:
                totals_any = True
                for k in totals:
                    totals[k] += float(macros.get(k, 0.0) or 0.0)

            items.append(
                FoodItemResult(
                    item_id=i,
                    food_name=food_name,
                    confidence=float(conf),
                    bbox_xyxy=[float(x) for x in bbox],
                    volume_ml=volume_ml,
                    grams_estimated=grams,
                    macros=macros,
                    usda_description=usda_desc,
                )
            )

        return {
            "image": image_id,
            "num_items": len(items),
            "items": [i.__dict__ for i in items],
            "totals": totals if totals_any else None,
        }

