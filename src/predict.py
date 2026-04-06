import sys
import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from utils import load_and_preprocess

# Paths
MODEL_PATH = "../model/animal_model.h5"
CLASS_PATH = "../model/classes.json"

# Load model
model = load_model(MODEL_PATH)

# Load class names (IMPORTANT)
if not os.path.exists(CLASS_PATH):
    print("❌ Error: classes.json not found. Run training first.")
    exit()

with open(CLASS_PATH, "r") as f:
    class_indices = json.load(f)

# Convert dictionary to ordered list
CLASS_NAMES = list(class_indices.keys())

def predict_image(img_path):
    """
    Predict animal from image path
    """

    # Check file exists
    if not os.path.exists(img_path):
        print("Error: Image path not found")
        return

    try:
        # Preprocess image
        img = load_and_preprocess(img_path)

        # Predict
        predictions = model.predict(img)
        predicted_index = np.argmax(predictions)
        confidence = np.max(predictions)

        predicted_class = CLASS_NAMES[predicted_index]

        # Output result
        print(f"✅ Predicted Animal: {predicted_class}")
        print(f"📊 Confidence: {confidence*100:.2f}%")

    except Exception as e:
        print("Error processing image:", e)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
    else:
        image_path = sys.argv[1]
        predict_image(image_path)