"""
test.py - Prediction function for hidden dataset evaluation
Material Stream Identification System
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageOps
import joblib
import tensorflow as tf
import pandas as pd
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input

# ===================== CONSTANTS =====================

CLASS_TO_ID = {
    'glass': 0,
    'paper': 1,
    'cardboard': 2,
    'plastic': 3,
    'metal': 4,
    'trash': 5,
    'Unknown': 6
}

ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}

IMG_SIZE = (128, 128)
MIN_SIZE = 32
CONFIDENCE_THRESHOLD = 0.6


# ===================== IMAGE PROCESSING =====================

def _load_and_process_image(img_path, img_name):
    """Load and preprocess a single image."""
    if os.path.getsize(img_path) == 0:
        print(f"Skipping empty file: {img_name}")
        return None

    pil_img = Image.open(img_path)
    pil_img = ImageOps.exif_transpose(pil_img)
    pil_img = pil_img.convert("RGB")
    img = np.array(pil_img)

    if img.shape[0] < MIN_SIZE or img.shape[1] < MIN_SIZE:
        print(f"Skipping small image: {img_name}")
        return None

    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32)
    return img


def _get_image_files(folder_path):
    """List all valid image files in a directory."""
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    return sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(valid_extensions)
    ])


# ===================== PREDICTION =====================

def _make_predictions(model, features_scaled):
    """Generate predictions with confidence threshold."""
    predictions = []

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features_scaled)
        max_probs = np.max(probs, axis=1)
        pred_classes = model.classes_[np.argmax(probs, axis=1)]

        for cls, prob in zip(pred_classes, max_probs):
            if prob < CONFIDENCE_THRESHOLD:
                predictions.append(6)  # Unknown
            else:
                predictions.append(CLASS_TO_ID.get(cls, 6))
    else:
        pred_classes = model.predict(features_scaled)
        predictions = [CLASS_TO_ID.get(cls, 6) for cls in pred_classes]

    return predictions


# ===================== MAIN FUNCTION =====================

def predict(data_folder_path, model_path, output_excel_path="predictions.xlsx"):
    """
    Predict material classes for all images in a folder
    and save results to an Excel file.
    """

    print(f"Loading model: {model_path}")
    model = joblib.load(model_path)

    scaler_path = os.path.join(os.path.dirname(model_path), "trained_scaler.pkl")
    scaler = joblib.load(scaler_path)

    print("Initializing feature extractor...")
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(128, 128, 3)
    )
    feature_extractor = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D()
    ])

    image_files = _get_image_files(data_folder_path)
    print(f"Found {len(image_files)} images")

    images = []
    valid_image_names = []

    for img_name in image_files:
        img_path = os.path.join(data_folder_path, img_name)
        try:
            img = _load_and_process_image(img_path, img_name)
            if img is not None:
                images.append(img)
                valid_image_names.append(img_name)
        except Exception as e:
            print(f"Error loading {img_name}: {e}")

    if len(images) == 0:
        raise RuntimeError("No valid images found!")

    images = np.array(images)

    print("Extracting features...")
    images = preprocess_input(images)
    features = feature_extractor.predict(images, batch_size=32, verbose=1)

    print("Scaling features...")
    features_scaled = scaler.transform(features)

    print("Making predictions...")
    predicted_ids = _make_predictions(model, features_scaled)
    predicted_labels = [ID_TO_CLASS[p] for p in predicted_ids]

    # ===================== SAVE TO EXCEL =====================

    df = pd.DataFrame({
        "Image_Name": valid_image_names,
        "Predicted_Label": predicted_labels
    })

    df.to_excel(output_excel_path, index=False)
    print(f"Predictions saved to: {output_excel_path}")

    return df


# ===================== ENTRY POINT =====================

if __name__ == "__main__":

    test_data_path = "path_to_hidden_test_folder"
    model_path = "trained_svm_model.pkl"

    try:
        predict(
            data_folder_path=test_data_path,
            model_path=model_path,
            output_excel_path="submission_predictions.xlsx"
        )
    except Exception as e:
        print("Error occurred:", e)
