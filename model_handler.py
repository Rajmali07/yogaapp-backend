import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)


class YogaModelHandler:
    """Handles loading and inference of the yoga pose model"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.input_size = (224, 224)  # Standard size for EfficientNet
        
        # Yoga pose classes - actual trained poses
        self.pose_classes = [
            "Anantasana",
            "Ardhakati Chakrasana",
            "Bhujangasana",
            "Kati Chakrasana",
            "Marjariasana",
            "Parvatasana",
            "Sarvangasana",
            "Tadasana",
            "Vajrasana",
            "Viparita Karani"
        ]
    
    def load_model(self):
        """Load TensorFlow SavedModel"""
        try:
            import tensorflow as tf
            logger.info(f"Loading SavedModel from {self.model_path}")
            self.model = tf.saved_model.load(self.model_path)
            logger.info("SavedModel loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def preprocess_image(self, image):
        """
        Preprocess image for model input
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            Preprocessed tensor ready for model
        """
        # Resize to model input size
        img_resized = cv2.resize(image, self.input_size)
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] - use float16 for FP16 model
        img_normalized = img_rgb.astype(np.float16) / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch
    
    def predict(self, image):
        """
        Run inference on a single image
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            import tensorflow as tf
            
            # Preprocess image
            input_tensor = self.preprocess_image(image)
            
            # Run inference
            infer = self.model.signatures["serving_default"]
            
            # Get input tensor name
            input_name = list(infer.structured_input_signature[1].keys())[0]
            predictions = infer(**{input_name: tf.constant(input_tensor)})
            
            # Get output
            output_key = list(predictions.keys())[0]
            output = predictions[output_key].numpy()[0]
            
            # Get predicted class and confidence
            predicted_idx = np.argmax(output)
            confidence = float(output[predicted_idx])
            
            # Get pose class name
            pose_class = self.pose_classes[predicted_idx] if predicted_idx < len(self.pose_classes) else f"Pose {predicted_idx}"
            
            # Determine if pose is correct (confidence threshold)
            is_correct = confidence > 0.75  # Adjust threshold as needed
            
            # Generate feedback
            feedback = self._generate_feedback(pose_class, confidence, is_correct)
            
            return {
                "pose_class": pose_class,
                "confidence": float(confidence),
                "is_correct": is_correct,
                "all_probabilities": {
                    self.pose_classes[i]: float(output[i]) 
                    for i in range(min(len(output), len(self.pose_classes)))
                },
                "feedback": feedback
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {
                "pose_class": "Unknown",
                "confidence": 0.0,
                "is_correct": False,
                "feedback": f"Error analyzing pose: {str(e)}"
            }
    
    def _generate_feedback(self, pose_class, confidence, is_correct):
        """Generate human-readable feedback"""
        if is_correct:
            if confidence > 0.9:
                return f"Perfect {pose_class}! Excellent form."
            else:
                return f"Good {pose_class}. Minor improvements possible."
        else:
            if confidence < 0.5:
                return f"Unable to clearly identify pose. Try adjusting position."
            else:
                return f"Detected {pose_class} but form needs adjustment. Check alignment and positioning."


# For testing
if __name__ == "__main__":
    # Test model loading
    handler = YogaModelHandler("D:/yoga_app/model_prep/yoga_savedmodel")
    handler.load_model()
    print("SavedModel loaded successfully!")
