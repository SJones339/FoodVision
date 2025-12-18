import os
from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image

from src.pipeline.yolo_midas_usda import FoodVisionWebPipeline


@st.cache_resource
def get_pipeline():
    # Load API key from environment (optionally via .env if user runs `source`/dotenv externally).
    usda_api_key = os.getenv("USDA_API_KEY")
    return FoodVisionWebPipeline(
        yolo_model_path="models/yolov8m_food_best.pt",
        usda_api_key=usda_api_key,
        enable_depth=True,
    )


def main():
    st.set_page_config(page_title="FoodVision", layout="centered")
    st.title("FoodVision: upload a meal photo")
    st.write("Upload an image → YOLO segments + labels → MiDaS depth → USDA macros.")

    if not os.getenv("USDA_API_KEY"):
        st.warning("USDA_API_KEY not set. The app will still detect foods, but macros will be blank.")

    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
    if not uploaded:
        return

    img_bytes = uploaded.getvalue()
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    with st.spinner("Running pipeline… (first run may download MiDaS weights)"):
        pipeline = get_pipeline()
        if getattr(pipeline, "depth_init_error", None):
            st.warning(f"Depth is disabled: {pipeline.depth_init_error}")
        results = pipeline.process_image(img)

    items = results.get("items", [])
    if not items:
        st.info("No food items detected.")
        return

    rows = []
    for it in items:
        macros = it.get("macros") or {}
        rows.append(
            {
                "item_id": it.get("item_id"),
                "food": it.get("food_name"),
                "confidence": it.get("confidence"),
                "volume_ml": it.get("volume_ml"),
                "grams_estimated": it.get("grams_estimated"),
                "cal": macros.get("cal"),
                "protein_g": macros.get("protein"),
                "carbs_g": macros.get("carbs"),
                "fat_g": macros.get("fat"),
            }
        )

    df = pd.DataFrame(rows)
    st.subheader("Detected items")
    st.dataframe(df, use_container_width=True)

    totals = results.get("totals")
    st.subheader("Meal totals")
    if totals:
        st.write(
            {
                "cal": totals.get("cal"),
                "protein_g": totals.get("protein"),
                "carbs_g": totals.get("carbs"),
                "fat_g": totals.get("fat"),
            }
        )
    else:
        st.info("Totals unavailable (USDA lookup missing/failed).")


if __name__ == "__main__":
    main()

