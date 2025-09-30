#!/usr/bin/env python3
"""
Quick training script for cattle breed classification
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import cv2
import json

def load_and_train():
    """Load dataset and train model quickly"""
    print("Loading dataset...")
    
    images = []
    labels = []
    breed_to_index = {}
    
    # Get breed folders
    dataset_path = 'dataset'
    breed_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    breed_folders.sort()
    
    print(f"Found {len(breed_folders)} breed categories")
    
    # Load images
    for i, folder in enumerate(breed_folders):
        breed_to_index[folder] = i
        folder_path = os.path.join(dataset_path, folder)
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        
        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            try:
                img = cv2.imread(image_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (224, 224))
                    img = img.astype(np.float32) / 255.0
                    images.append(img)
                    labels.append(i)
            except:
                pass
    
    if len(images) == 0:
        print("No images found!")
        return
    
    X = np.array(images)
    y = tf.keras.utils.to_categorical(labels, num_classes=len(breed_folders))
    
    print(f"Loaded {len(X)} images")
    
    # Create simple model
    model = Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(breed_folders), activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train model
    print("Training model...")
    model.fit(X, y, epochs=20, batch_size=4, validation_split=0.2, verbose=1)
    
    # Save model
    os.makedirs('model', exist_ok=True)
    model.save('model/cattle_breed_model.h5')
    
    # Save breed names
    with open('model/breed_names.json', 'w') as f:
        json.dump(breed_folders, f)
    
    print("Model trained and saved successfully!")
    print(f"Breed names: {breed_folders}")

if __name__ == "__main__":
    load_and_train()
