#!/usr/bin/env python3
"""
Train model using transfer learning for better accuracy with limited data
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import json
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_dataset(dataset_path='dataset'):
    """Load images from the dataset folders"""
    print("Loading dataset from:", dataset_path)
    
    images = []
    labels = []
    breed_to_index = {}
    
    # Get all breed folders
    breed_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    breed_folders.sort()
    
    print(f"Found {len(breed_folders)} breed categories:")
    for i, folder in enumerate(breed_folders):
        print(f"  {i}: {folder}")
        breed_to_index[folder] = i
    
    # Load images from each breed folder
    for breed_folder in breed_folders:
        folder_path = os.path.join(dataset_path, breed_folder)
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        
        print(f"Loading {len(image_files)} images from {breed_folder}")
        
        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            try:
                # Load and preprocess image
                img = cv2.imread(image_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (224, 224))
                    img = img.astype(np.float32) / 255.0
                    
                    images.append(img)
                    labels.append(breed_to_index[breed_folder])
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
    
    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)
    
    # Convert labels to categorical
    y_categorical = tf.keras.utils.to_categorical(y, num_classes=len(breed_folders))
    
    print(f"Loaded {len(X)} images with {len(breed_folders)} classes")
    print(f"Image shape: {X.shape}")
    print(f"Label shape: {y_categorical.shape}")
    
    return X, y_categorical, breed_folders

def create_transfer_learning_model(num_classes):
    """Create model using transfer learning with MobileNetV2"""
    print("Creating transfer learning model...")
    
    # Load pre-trained MobileNetV2 model
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Add custom layers
    model = tf.keras.Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Transfer learning model created successfully!")
    model.summary()
    
    return model

def train_model(X, y, breed_names):
    """Train the model with data augmentation"""
    print("Starting model training...")
    
    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1)
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    # Create data generators for augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator()
    
    # Train the model
    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=4),
        steps_per_epoch=max(1, len(X_train) // 4),
        epochs=30,
        validation_data=val_datagen.flow(X_val, y_val, batch_size=4),
        validation_steps=max(1, len(X_val) // 4),
        verbose=1
    )
    
    return history

def main():
    """Main training function"""
    print("=" * 60)
    print("Training Cattle Breed Classification Model (Transfer Learning)")
    print("=" * 60)
    
    # Load dataset
    X, y, breed_names = load_dataset()
    
    if len(X) == 0:
        print("No images found in dataset! Please check your dataset folder.")
        return
    
    # Create model
    model = create_transfer_learning_model(num_classes=len(breed_names))
    
    # Train model
    history = train_model(X, y, breed_names)
    
    # Save model
    model_path = 'model/cattle_breed_model.h5'
    model.save(model_path)
    print(f"Model saved as: {model_path}")
    
    # Save breed names
    with open('model/breed_names.json', 'w') as f:
        json.dump(breed_names, f)
    print("Breed names saved as: model/breed_names.json")
    
    # Print final accuracy
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Final validation accuracy: {val_accuracy:.4f}")
    
    print("Training completed successfully!")
    return model, breed_names

if __name__ == "__main__":
    model, breed_names = main()
