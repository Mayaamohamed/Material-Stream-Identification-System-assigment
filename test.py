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
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input


def _load_and_process_image(img_path, img_name, img_size, min_size):
    """Load and preprocess a single image."""
    if os.path.getsize(img_path) == 0:
        print(f"Skipping empty file: {img_name}")
        return None
    
    pil_img = Image.open(img_path)
    pil_img = ImageOps.exif_transpose(pil_img)
    pil_img = pil_img.convert("RGB")
    img = np.array(pil_img)
    
    if img.shape[0] < min_size or img.shape[1] < min_size:
        print(f"Skipping small image: {img_name}")
        return None
    
    img = cv2.resize(img, img_size)
    img = img.astype(np.float32) / 255.0
    return img


def _get_image_files(data_file_path):
    """Get list of valid image files from directory."""
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = []
    for filename in sorted(os.listdir(data_file_path)):
        if filename.lower().endswith(valid_extensions):
            image_files.append(filename)
    return image_files


def _make_predictions(model, x_features_scaled, confidence_threshold):
    """Generate predictions from features."""
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(x_features_scaled)
        max_probs = np.max(probabilities, axis=1)
        pred_classes = model.classes_[np.argmax(probabilities, axis=1)]
        return ["Unknown" if prob < confidence_threshold else cls 
                for cls, prob in zip(pred_classes, max_probs)]
    else:
        return list(model.predict(x_features_scaled))


def predict(data_file_path, best_model_path):
    """
    Predict material classes for images in the given folder.
    
    Parameters:
    -----------
    data_file_path : str
        Path to folder containing test images
    best_model_path : str
        Path to the trained model file (.pkl)
    
    Returns:
    --------
    predictions : list
        List of predicted class labels for each image
    """
    
   
    IMG_SIZE = (128, 128)
    MIN_SIZE = 32
    CONFIDENCE_THRESHOLD = 0.6
    
    
    print(f"Loading model from: {best_model_path}")
    model = joblib.load(best_model_path)
    scaler_path = os.path.join(os.path.dirname(best_model_path), "trained_scaler.pkl")
    scaler = joblib.load(scaler_path)
    
  
    print("Initializing feature extractor...")
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
    feature_extractor = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D()
    ])
    
   
    print(f"Loading images from: {data_file_path}")
    image_files = _get_image_files(data_file_path)
    print(f"Found {len(image_files)} images")
    

    X_test = []
    for img_name in image_files:
        img_path = os.path.join(data_file_path, img_name)
        try:
            img = _load_and_process_image(img_path, img_name, IMG_SIZE, MIN_SIZE)
            if img is not None:
                X_test.append(img)
        except Exception as e:
            print(f"Error loading {img_name}: {e}")
    
    print(f"Successfully loaded {len(X_test)} images")
    
    if len(X_test) == 0:
        print("No valid images found!")
        return []
    
 
    X_test = np.array(X_test)
    

    print("Extracting features...")
    x_test_preprocessed = preprocess_input(X_test * 255.0)
    x_features = feature_extractor.predict(x_test_preprocessed, batch_size=32, verbose=1)
  
    print("Scaling features...")
    x_features_scaled = scaler.transform(x_features)
    
    print("Making predictions...")
    predictions = _make_predictions(model, x_features_scaled, CONFIDENCE_THRESHOLD)
    
    print(f"Generated {len(predictions)} predictions")
    print(f"Prediction distribution: {dict(zip(*np.unique(predictions, return_counts=True)))}")
    
    return predictions



if __name__ == "__main__":
   
    test_data_path = "C:/Users/roqai/Downloads/dataset/glass"  
    model_path = "C:/Users/roqai/Downloads/trained_svm_model.pkl"     
    
    try:
        predictions = predict(test_data_path, model_path)
        print("\nPredictions:")
        for i, pred in enumerate(predictions):
            print(f"Image {i+1}: {pred}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()