import cv2
import numpy as np
import joblib
import tensorflow as tf
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
import time

# Configuration
MODEL_PATH = "C:/Users/roqai/Downloads/trained_svm_model.pkl"  # Change to your model path
SCALER_PATH = "C:/Users/roqai/Downloads/trained_scaler.pkl"
CONFIDENCE_THRESHOLD = 0.6

# Material class names and colors for display
CLASS_NAMES = {
    0: "Glass",
    1: "Paper", 
    2: "Cardboard",
    3: "Plastic",
    4: "Metal",
    5: "Trash",
    6: "Unknown"
}

CLASS_COLORS = {
    "Glass": (255, 200, 100),
    "Paper": (255, 255, 255),
    "Cardboard": (139, 90, 43),
    "Plastic": (100, 150, 255),
    "Metal": (180, 180, 180),
    "Trash": (50, 50, 50),
    "Unknown": (128, 128, 128)
}

class MaterialClassifier:
    def __init__(self, model_path, scaler_path):
        print("Loading models...")
        
        # Load the trained classifier
        self.classifier = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        # Load MobileNetV2 for feature extraction
        base_model = MobileNetV2(weights="imagenet", include_top=False, 
                                input_shape=(128, 128, 3))
        self.feature_extractor = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D()
        ])
        
        print("Models loaded successfully!")
    
    def preprocess_frame(self, frame):
        """Preprocess camera frame for classification"""
        # Resize to 128x128
        img = cv2.resize(frame, (128, 128))
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        return img
    
    def extract_features(self, img):
        """Extract features using MobileNetV2"""
        # Preprocess for MobileNetV2
        img_preprocessed = preprocess_input(img * 255.0)
        img_batch = np.expand_dims(img_preprocessed, axis=0)
        
        # Extract features
        features = self.feature_extractor.predict(img_batch, verbose=0)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        return features_scaled
    
    def predict_with_rejection(self, features):
        """Predict class with confidence threshold"""
        probs = self.classifier.predict_proba(features)
        max_prob = np.max(probs, axis=1)[0]
        pred_class = self.classifier.classes_[np.argmax(probs, axis=1)][0]
        
        if max_prob < CONFIDENCE_THRESHOLD:
            pred_class = "Unknown"
        
        return pred_class, max_prob

def draw_prediction_box(frame, class_name, confidence):
    """Draw prediction information on frame"""
    h, w = frame.shape[:2]
    
    # Create semi-transparent overlay
    overlay = frame.copy()
    
    # Draw background rectangle
    cv2.rectangle(overlay, (10, 10), (w - 10, 120), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    
    # Get color for class
    color = CLASS_COLORS.get(class_name, (255, 255, 255))
    
    # Draw class name
    cv2.putText(frame, f"Material: {class_name}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    
    # Draw confidence
    conf_text = f"Confidence: {confidence:.2%}"
    cv2.putText(frame, conf_text, (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Draw confidence bar
    bar_width = int((w - 40) * confidence)
    cv2.rectangle(frame, (20, 100), (20 + bar_width, 110), color, -1)
    cv2.rectangle(frame, (20, 100), (w - 20, 110), (255, 255, 255), 2)
    
    return frame

def main():
    # Initialize classifier
    try:
        classifier = MaterialClassifier(MODEL_PATH, SCALER_PATH)
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please update MODEL_PATH and SCALER_PATH to point to your model files.")
        return
    
    # Open camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("\n=== Material Stream Identification System ===")
    print("Camera started. Press 'q' to quit.")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print("\nPoint camera at recyclable materials...")
    
    # Variables for FPS calculation
    fps = 0
    frame_count = 0
    start_time = time.time()
    
    # Main loop
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Process every frame
        try:
            # Preprocess frame
            processed_img = classifier.preprocess_frame(frame)
            
            # Extract features
            features = classifier.extract_features(processed_img)
            
            # Predict
            class_name, confidence = classifier.predict_with_rejection(features)
            
            # Draw prediction on frame
            frame = draw_prediction_box(frame, class_name, confidence)
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 10 == 0:
                end_time = time.time()
                fps = 10 / (end_time - start_time)
                start_time = time.time()
            
            # Draw FPS
            cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            cv2.putText(frame, "Processing Error", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display frame
        cv2.imshow('Material Stream Identification System', frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\nApplication closed.")

if __name__ == "__main__":
    main()