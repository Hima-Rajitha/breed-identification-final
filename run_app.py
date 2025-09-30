#!/usr/bin/env python3
"""
Cattle Breed Classification Web Application
This script trains the model (if needed) and starts the Flask web application.
"""

import os
import sys
import subprocess

def check_model_exists():
    """Check if the trained model exists"""
    model_path = 'model/cattle_breed_model.h5'
    breed_names_path = 'model/breed_names.json'
    return os.path.exists(model_path) and os.path.exists(breed_names_path)

def train_model():
    """Train the CNN model for breed classification"""
    print("Training CNN model for cattle breed classification...")
    print("This may take a few minutes...")
    
    # Change to model directory
    os.chdir('model')
    
    try:
        # Run the model training script
        result = subprocess.run([sys.executable, 'create_simple_model.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Model training completed successfully!")
            print(result.stdout)
        else:
            print("Error during model training:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"Error running model training: {e}")
        return False
    finally:
        # Change back to root directory
        os.chdir('..')
    
    return True

def start_flask_app():
    """Start the Flask web application"""
    print("Starting Flask web application...")
    print("Open your browser and go to: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    
    try:
        # Import and run the Flask app
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting Flask app: {e}")

def main():
    """Main function to orchestrate the application startup"""
    print("=" * 60)
    print("Cattle Breed Classification Web Application")
    print("=" * 60)
    
    # Check if model exists
    if not check_model_exists():
        print("Trained model not found. Training model...")
        if not train_model():
            print("Failed to train model. Please check the dataset and try again.")
            return
    else:
        print("Trained model found. Skipping training.")
    
    # Start the Flask application
    start_flask_app()

if __name__ == "__main__":
    main()
