#!/usr/bin/env python3
"""
Create a simple working model for the cattle breed classification
"""

import os
import tensorflow as tf
import json
import numpy as np

def create_simple_model():
    """Create a simple model that can be loaded by the Flask app"""
    print("Creating simple cattle breed classification model...")
    
    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    
    # Create a simple model architecture
    model = tf.keras.Sequential([
        # Input layer
        tf.keras.layers.Input(shape=(224, 224, 3)),
        
        # Convolutional layers
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Flatten and dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(24, activation='softmax')  # 24 breeds
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create dummy weights (this is just to make the model loadable)
    dummy_input = np.random.random((1, 224, 224, 3))
    _ = model(dummy_input)
    
    # Save the model
    model.save('model/cattle_breed_model.h5')
    print("✓ Model saved successfully")
    
    # Create breed names file
    breed_names = [
        'Jersey cross', 'Ayrshire', 'Brown Swiss', 'Guernsey', 'Holstein Friesian',
        'Jersey', 'Red Dane', 'Gir', 'Red Sindhi', 'Sahiwal',
        'Alambadi', 'Amritmahal', 'Bargur', 'Hallikar', 'Kangayam',
        'Khillari', 'Pulikulam', 'Umblachery', 'Deoni', 'Hariana',
        'Kankrej', 'Krishna Valley', 'Ongole', 'Tharparkar'
    ]
    
    with open('model/breed_names.json', 'w') as f:
        json.dump(breed_names, f)
    print("✓ Breed names saved successfully")
    
    print("Model creation completed!")
    return model

if __name__ == "__main__":
    create_simple_model()

