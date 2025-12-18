import cv2
import numpy as np
import joblib
import tensorflow as tf
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
import time


MODEL_PATH = "C:/Users/roqai/Downloads/trained_svm_model.pkl" #change to your own path
SCALER_PATH = "C:/Users/roqai/Downloads/trained_scaler.pkl"   #change to your own path
CONFIDENCE_THRESHOLD = 0.6


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
        
      
        self.classifier = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
      
        base_model = MobileNetV2(weights="imagenet", include_top=False, 
                                input_shape=(128, 128, 3))
        self.feature_extractor = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D()
        ])
        
        print("Models loaded successfully!")
    
    def preprocess_frame(self, frame):
        """Preprocess camera frame for classification"""
      
        img = cv2.resize(frame, (128, 128))
        
      
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
     
        img = img.astype(np.float32) / 255.0
        
        return img
    
    def extract_features(self, img):
        """Extract features using MobileNetV2"""
    
        img_preprocessed = preprocess_input(img * 255.0)
        img_batch = np.expand_dims(img_preprocessed, axis=0)
        
      
        features = self.feature_extractor.predict(img_batch, verbose=0)
        
  
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
    _, w = frame.shape[:2]
    
   
    overlay = frame.copy()
    
   
    cv2.rectangle(overlay, (10, 10), (w - 10, 120), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    
   
    color = CLASS_COLORS.get(class_name, (255, 255, 255))
    
    
    cv2.putText(frame, f"Material: {class_name}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    
  
    conf_text = f"Confidence: {confidence:.2%}"
    cv2.putText(frame, conf_text, (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
   
    bar_width = int((w - 40) * confidence)
    cv2.rectangle(frame, (20, 100), (20 + bar_width, 110), color, -1)
    cv2.rectangle(frame, (20, 100), (w - 20, 110), (255, 255, 255), 2)
    
    return frame

def main():
  
    try:
        classifier = MaterialClassifier(MODEL_PATH, SCALER_PATH)
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please update MODEL_PATH and SCALER_PATH to point to your model files.")
        return
    
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
   
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("\n=== Material Stream Identification System ===")
    print("Camera started. Press 'q' to quit.")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print("\nPoint camera at recyclable materials...")
    
   
    fps = 0
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            break
        
       
        try:
          
            processed_img = classifier.preprocess_frame(frame)
            
            
            features = classifier.extract_features(processed_img)
            
           
            class_name, confidence = classifier.predict_with_rejection(features)
            
          
            frame = draw_prediction_box(frame, class_name, confidence)
            

            frame_count += 1
            if frame_count % 10 == 0:
                end_time = time.time()
                fps = 10 / (end_time - start_time)
                start_time = time.time()
            
           
            cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            cv2.putText(frame, "Processing Error", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
      
        cv2.imshow('Material Stream Identification System', frame)
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nApplication closed.")

if __name__ == "__main__":
    main()