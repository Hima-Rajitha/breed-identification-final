#!/usr/bin/env python3
"""
Train model on individual breed names instead of categories
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import cv2
import json

def load_individual_breeds():
    """Load dataset with individual breed names"""
    print("Loading dataset with individual breed names...")
    
    images = []
    labels = []
    breed_to_index = {}
    
    # Define individual breeds and their categories
    breed_mapping = {
        # Cross-bred Dairy Cattle
        'Cross_bred-Dairy_Cattle': ['Jersey cross'],
        
        # Exotic dairy breeds of cattle
        'Exotic dairy breeds of cattle': ['Ayrshire', 'Brown Swiss', 'Guernsey', 'Holstein Friesian', 'Jersey', 'Red Dane'],
        
        # Indigenous_dairy_breeds_of_cattle
        'Indigenous_dairy_breeds_of_cattle': ['Gir', 'Red Sindhi', 'Sahiwal'],
        
        # Indigenous_Draught_breeds_of_cattle
        'Indigenous_Draught_breeds_of_cattle': ['Alambadi', 'Amritmahal', 'Bargur', 'Hallikar', 'Kangayam', 'Khillari', 'Pulikulam', 'Umblachery'],
        
        # Indigenous_Dual_purpose_breeds_of_Cattle
        'Indigenous_Dual_purpose_breeds_of_Cattle': ['Deoni', 'Hariana', 'Kankrej', 'Krishna Valley', 'Ongole', 'Tharparkar']
    }
    
    # Create breed to index mapping
    all_breeds = []
    for category, breeds in breed_mapping.items():
        all_breeds.extend(breeds)
    
    for i, breed in enumerate(all_breeds):
        breed_to_index[breed] = i
    
    print(f"Total breeds: {len(all_breeds)}")
    print("Breeds:", all_breeds)
    
    # Load images from dataset
    dataset_path = 'dataset'
    for category_folder in os.listdir(dataset_path):
        if not os.path.isdir(os.path.join(dataset_path, category_folder)):
            continue
            
        category_path = os.path.join(dataset_path, category_folder)
        image_files = [f for f in os.listdir(category_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        
        print(f"Processing {category_folder}: {len(image_files)} images")
        
        for image_file in image_files:
            image_path = os.path.join(category_path, image_file)
            
            # Determine breed from filename
            breed_name = None
            filename_lower = image_file.lower()
            
            # Map filename to breed
            for breed in all_breeds:
                if breed.lower().replace(' ', '_') in filename_lower or breed.lower() in filename_lower:
                    breed_name = breed
                    break
            
            # If no specific breed found, use first breed from category
            if breed_name is None and category_folder in breed_mapping:
                breed_name = breed_mapping[category_folder][0]
            
            if breed_name and breed_name in breed_to_index:
                try:
                    img = cv2.imread(image_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (224, 224))
                        img = img.astype(np.float32) / 255.0
                        images.append(img)
                        labels.append(breed_to_index[breed_name])
                        print(f"  Loaded {image_file} as {breed_name}")
                except Exception as e:
                    print(f"Error loading {image_path}: {e}")
    
    if len(images) == 0:
        print("No images found!")
        return None, None, None
    
    X = np.array(images)
    y = tf.keras.utils.to_categorical(labels, num_classes=len(all_breeds))
    
    print(f"Loaded {len(X)} images for {len(all_breeds)} breeds")
    
    return X, y, all_breeds

def create_model(num_classes):
    """Create CNN model"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    """Main training function"""
    print("=" * 60)
    print("Training Individual Breed Classification Model")
    print("=" * 60)
    
    # Load dataset
    X, y, breed_names = load_individual_breeds()
    
    if X is None:
        print("Failed to load dataset!")
        return
    
    # Create model
    model = create_model(len(breed_names))
    
    # Train model
    print("Training model...")
    model.fit(X, y, epochs=30, batch_size=4, validation_split=0.2, verbose=1)
    
    # Save model
    os.makedirs('model', exist_ok=True)
    model.save('model/cattle_breed_model.h5')
    
    # Save breed names
    with open('model/breed_names.json', 'w') as f:
        json.dump(breed_names, f)
    
    print("Model trained and saved successfully!")
    print(f"Individual breeds: {breed_names}")

if __name__ == "__main__":
    main()
