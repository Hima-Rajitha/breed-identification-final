#!/usr/bin/env python3
"""
Test the trained model with sample images
"""

import os
import numpy as np
import tensorflow as tf
import cv2
import json

def test_model():
    """Test the trained model"""
    print("Testing the trained cattle breed classification model...")
    
    # Load model
    try:
        model = tf.keras.models.load_model('model/cattle_breed_model.h5')
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Load breed names
    try:
        with open('model/breed_names.json', 'r') as f:
            breed_names = json.load(f)
        print(f"✓ Loaded {len(breed_names)} breed names")
        print("Breeds:", breed_names)
    except Exception as e:
        print(f"✗ Error loading breed names: {e}")
        return
    
    # Test with a sample image
    dataset_path = 'dataset'
    test_images = []
    
    # Find a test image
    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)
        if os.path.isdir(category_path):
            for image_file in os.listdir(category_path):
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    test_images.append(os.path.join(category_path, image_file))
                    break
            if test_images:
                break
    
    if not test_images:
        print("No test images found!")
        return
    
    # Test prediction
    test_image_path = test_images[0]
    print(f"\nTesting with image: {test_image_path}")
    
    try:
        # Load and preprocess image
        img = cv2.imread(test_image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Make prediction
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        predicted_breed = breed_names[predicted_class]
        
        print(f"Predicted breed: {predicted_breed}")
        print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        
        # Show top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        print("\nTop 3 predictions:")
        for i, idx in enumerate(top_3_indices):
            breed = breed_names[idx]
            conf = predictions[0][idx]
            print(f"  {i+1}. {breed}: {conf:.4f} ({conf*100:.2f}%)")
        
    except Exception as e:
        print(f"Error testing model: {e}")

if __name__ == "__main__":
    test_model()
