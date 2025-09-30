#!/usr/bin/env python3
"""
Demo script to test the cattle breed classification system
"""

import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

def test_model_creation():
    """Test if we can create and save a simple model"""
    print("Testing model creation...")
    
    try:
        # Create a simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(25, activation='softmax')  # 25 breeds
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Create model directory if it doesn't exist
        os.makedirs('model', exist_ok=True)
        
        # Save model
        model.save('model/cattle_breed_model.h5')
        print("✓ Model created and saved successfully")
        
        # Create dummy breed names
        breed_names = [
            'Jersey cross', 'Ayrshire', 'Brown Swiss', 'Guernsey', 'Holstein Friesian',
            'Jersey', 'Red Dane', 'Gir', 'Red Sindhi', 'Sahiwal',
            'Alambadi', 'Amritmahal', 'Bargur', 'Hallikar', 'Kangayam',
            'Khillari', 'Pulikulam', 'Umblachery', 'Deoni', 'Hariana',
            'Kankrej', 'Krishna Valley', 'Ongole', 'Tharparkar', 'Unknown'
        ]
        
        import json
        with open('model/breed_names.json', 'w') as f:
            json.dump(breed_names, f)
        print("✓ Breed names saved successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False

def test_image_processing():
    """Test image processing functionality"""
    print("\nTesting image processing...")
    
    try:
        # Create a dummy image
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Test OpenCV processing
        img_cv = cv2.cvtColor(dummy_image, cv2.COLOR_RGB2BGR)
        img_resized = cv2.resize(img_cv, (224, 224))
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        print("✓ OpenCV image processing works")
        
        # Test PIL processing
        img_pil = Image.fromarray(dummy_image)
        img_pil_resized = img_pil.resize((224, 224))
        
        print("✓ PIL image processing works")
        
        return True
        
    except Exception as e:
        print(f"✗ Image processing failed: {e}")
        return False

def test_flask_imports():
    """Test Flask-related imports"""
    print("\nTesting Flask imports...")
    
    try:
        from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_from_directory
        from werkzeug.security import generate_password_hash, check_password_hash
        from werkzeug.utils import secure_filename
        import sqlite3
        
        print("✓ All Flask imports successful")
        return True
        
    except Exception as e:
        print(f"✗ Flask imports failed: {e}")
        return False

def create_demo_data():
    """Create some demo data for testing"""
    print("\nCreating demo data...")
    
    try:
        # Create uploads directory
        os.makedirs('uploads', exist_ok=True)
        print("✓ Uploads directory created")
        
        # Create a simple demo image
        demo_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        cv2.imwrite('uploads/demo_image.jpg', cv2.cvtColor(demo_image, cv2.COLOR_RGB2BGR))
        print("✓ Demo image created")
        
        return True
        
    except Exception as e:
        print(f"✗ Demo data creation failed: {e}")
        return False

def main():
    """Run demo tests"""
    print("=" * 60)
    print("Cattle Breed Classification - Demo Test")
    print("=" * 60)
    
    tests = [
        ("Model Creation", test_model_creation),
        ("Image Processing", test_image_processing),
        ("Flask Imports", test_flask_imports),
        ("Demo Data", create_demo_data)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("DEMO TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 Demo tests passed! The system is ready to run.")
        print("You can now run 'python run_app.py' to start the web application.")
    else:
        print("\n⚠ Some demo tests failed. Please check the error messages above.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
