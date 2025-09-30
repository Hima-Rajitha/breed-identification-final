from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import sqlite3
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import json
from datetime import datetime
import re

app = Flask(__name__, template_folder='frontend/templates', static_folder='frontend/static')
app.secret_key = 'your-secret-key-change-this-in-production'

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Database initialization
def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Load breed information
def load_breed_info():
    breed_info = {
        # Cross-bred Dairy Cattle
        'Jersey cross': {
            'category': 'Cross-bred Dairy Cattle',
            'features': [
                'High milk production with good butterfat content',
                'Adaptable to various climatic conditions',
                'Medium-sized with distinctive brown coloring',
                'Excellent feed conversion efficiency',
                'Good temperament and easy to handle'
            ]
        },
        
        # Exotic Dairy Breeds
        'Ayrshire': {
            'category': 'Exotic Dairy Breeds',
            'features': [
                'Strong and hardy breed with good longevity',
                'Excellent milk production with moderate butterfat',
                'Red and white spotted coat pattern',
                'Good foraging ability and disease resistance',
                'Well-suited for grazing systems'
            ]
        },
        'Brown Swiss': {
            'category': 'Exotic Dairy Breeds',
            'features': [
                'Large-sized breed with excellent milk production',
                'High protein content in milk',
                'Solid brown color with lighter muzzle',
                'Calm temperament and good mothering ability',
                'Adaptable to various management systems'
            ]
        },
        'Guernsey': {
            'category': 'Exotic Dairy Breeds',
            'features': [
                'Golden milk with high butterfat content',
                'Medium-sized with fawn and white coloring',
                'Efficient feed conversion and good temperament',
                'Excellent for small-scale dairy operations',
                'Known for rich, creamy milk quality'
            ]
        },
        'Holstein Friesian': {
            'category': 'Exotic Dairy Breeds',
            'features': [
                'World\'s highest milk producing breed',
                'Distinctive black and white spotted pattern',
                'Large frame with excellent feed efficiency',
                'Most popular dairy breed globally',
                'High volume milk production with moderate butterfat'
            ]
        },
        'Jersey': {
            'category': 'Exotic Dairy Breeds',
            'features': [
                'Highest butterfat content in milk',
                'Small to medium-sized with fawn coloring',
                'Excellent feed efficiency and adaptability',
                'Calm temperament and easy calving',
                'Premium milk quality for dairy products'
            ]
        },
        'Red Dane': {
            'category': 'Exotic Dairy Breeds',
            'features': [
                'Solid red coloring with good milk production',
                'Hardy breed with good disease resistance',
                'Medium to large frame size',
                'Good temperament and mothering ability',
                'Well-suited for intensive dairy farming'
            ]
        },
        
        # Indigenous Dairy Breeds
        'Gir': {
            'category': 'Indigenous Dairy Breeds',
            'features': [
                'Excellent milk production in tropical conditions',
                'Distinctive drooping ears and hump',
                'Red and white spotted coat pattern',
                'High disease resistance and adaptability',
                'Good for crossbreeding programs'
            ]
        },
        'Red Sindhi': {
            'category': 'Indigenous Dairy Breeds',
            'features': [
                'Solid red color with good milk yield',
                'Well-adapted to hot and humid climates',
                'Medium-sized with good temperament',
                'High butterfat content in milk',
                'Excellent for small-scale dairy farming'
            ]
        },
        'Sahiwal': {
            'category': 'Indigenous Dairy Breeds',
            'features': [
                'Best indigenous dairy breed of India',
                'Red color with white spots on face and legs',
                'High milk production with good butterfat',
                'Excellent heat tolerance and disease resistance',
                'Good for both milk and meat production'
            ]
        },
        
        # Indigenous Draught Breeds
        'Alambadi': {
            'category': 'Indigenous Draught Breeds',
            'features': [
                'Strong and sturdy build for heavy work',
                'Gray or white color with good endurance',
                'Excellent for agricultural operations',
                'Hardy and resistant to tropical diseases',
                'Good temperament for working animals'
            ]
        },
        'Amritmahal': {
            'category': 'Indigenous Draught Breeds',
            'features': [
                'Large-sized breed with great strength',
                'Gray color with distinctive appearance',
                'Excellent for heavy draught work',
                'Good endurance and disease resistance',
                'Well-suited for agricultural operations'
            ]
        },
        'Bargur': {
            'category': 'Indigenous Draught Breeds',
            'features': [
                'Medium-sized with good working ability',
                'Brown or gray coloring',
                'Excellent for light to medium work',
                'Good temperament and easy handling',
                'Well-adapted to local conditions'
            ]
        },
        'Hallikar': {
            'category': 'Indigenous Draught Breeds',
            'features': [
                'Medium-sized breed with good strength',
                'Gray or white color with compact build',
                'Excellent for agricultural work',
                'Good endurance and disease resistance',
                'Well-suited for draught purposes'
            ]
        },
        'Kangayam': {
            'category': 'Indigenous Draught Breeds',
            'features': [
                'Strong and muscular build',
                'White or gray color with good size',
                'Excellent for heavy agricultural work',
                'Good temperament and working ability',
                'Well-adapted to local farming conditions'
            ]
        },
        'Khillari': {
            'category': 'Indigenous Draught Breeds',
            'features': [
                'Medium-sized with good working capacity',
                'Gray or white color with compact frame',
                'Excellent for agricultural operations',
                'Good endurance and disease resistance',
                'Well-suited for draught work'
            ]
        },
        'Pulikulam': {
            'category': 'Indigenous Draught Breeds',
            'features': [
                'Small to medium-sized breed',
                'Gray or white color with good strength',
                'Excellent for light agricultural work',
                'Good temperament and easy handling',
                'Well-adapted to local conditions'
            ]
        },
        'Umblachery': {
            'category': 'Indigenous Draught Breeds',
            'features': [
                'Medium-sized with good working ability',
                'Gray color with compact build',
                'Excellent for agricultural operations',
                'Good endurance and disease resistance',
                'Well-suited for draught purposes'
            ]
        },
        
        # Indigenous Dual Purpose Breeds
        'Deoni': {
            'category': 'Indigenous Dual Purpose Breeds',
            'features': [
                'Good for both milk and draught work',
                'White color with black spots',
                'Medium-sized with good strength',
                'Excellent adaptability to local conditions',
                'Good temperament for dual purposes'
            ]
        },
        'Hariana': {
            'category': 'Indigenous Dual Purpose Breeds',
            'features': [
                'Excellent dual-purpose breed',
                'White color with good size',
                'Good milk production and working ability',
                'Well-adapted to northern Indian conditions',
                'Good temperament and disease resistance'
            ]
        },
        'Kankrej': {
            'category': 'Indigenous Dual Purpose Breeds',
            'features': [
                'Large-sized dual-purpose breed',
                'Silver-gray color with good strength',
                'Excellent for both milk and work',
                'Good adaptability to arid conditions',
                'Well-suited for farming operations'
            ]
        },
        'Krishna Valley': {
            'category': 'Indigenous Dual Purpose Breeds',
            'features': [
                'Good dual-purpose breed',
                'Gray or white color with medium size',
                'Good milk production and working ability',
                'Well-adapted to local conditions',
                'Good temperament for dual purposes'
            ]
        },
        'Ongole': {
            'category': 'Indigenous Dual Purpose Breeds',
            'features': [
                'Large-sized with excellent strength',
                'White color with distinctive appearance',
                'Good for both milk and heavy work',
                'Excellent adaptability to tropical conditions',
                'Well-suited for agricultural operations'
            ]
        },
        'Tharparkar': {
            'category': 'Indigenous Dual Purpose Breeds',
            'features': [
                'Excellent dual-purpose breed',
                'White color with good size',
                'Good milk production and working ability',
                'Well-adapted to desert conditions',
                'Good temperament and disease resistance'
            ]
        }
    }
    return breed_info

# Load the trained model (we'll create this)
def load_model():
    try:
        model = tf.keras.models.load_model('model/cattle_breed_model.h5')
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def load_breed_names():
    try:
        with open('model/breed_names.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading breed names: {e}")
        return []

# Initialize database and load data
init_db()
breed_info = load_breed_info()
model = load_model()
breed_names = load_breed_names()

def _normalize_breed_name(name: str) -> str:
    """Normalize breed names to compare irrespective of case/spacing/punctuation."""
    return re.sub(r"[^a-z0-9]", "", name.lower())

def resolve_breed_info(predicted_name: str):
    """Resolve predicted breed name to our `breed_info` keys using normalization.
    Returns (resolved_name, info_dict). If not found, returns default info.
    """
    normalized_target = _normalize_breed_name(predicted_name)
    for known_name in breed_info.keys():
        if _normalize_breed_name(known_name) == normalized_target:
            return known_name, breed_info[known_name]
    # Fallback when exact match not found
    return predicted_name, {
        'category': 'Unknown',
        'features': [
            'Feature data not available for this predicted breed.',
            'Please verify dataset folder naming matches breed names.',
            'Ensure spacing/case matches or add mapping in app.',
            'Add more images per breed to improve accuracy.',
            'Consider retraining with transfer learning.'
        ]
    }

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Preprocess image for model prediction"""
    try:
        # Load and resize image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_breed(image_path):
    """Predict breed from image"""
    if model is None:
        return None, 0.0
    
    try:
        # Preprocess image
        processed_img = preprocess_image(image_path)
        if processed_img is None:
            return None, 0.0
        
        # Make prediction
        predictions = model.predict(processed_img)
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        # Map class index to breed name
        if predicted_class < len(breed_names):
            predicted_breed = breed_names[predicted_class]
            return predicted_breed, confidence
        else:
            return None, 0.0
            
    except Exception as e:
        print(f"Error predicting breed: {e}")
        return None, 0.0

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        if not username or not email or not password:
            flash('All fields are required!', 'error')
            return render_template('register.html')
        
        # Hash password
        password_hash = generate_password_hash(password)
        
        # Save to database
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        
        try:
            cursor.execute('INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
                         (username, email, password_hash))
            conn.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists!', 'error')
        finally:
            conn.close()
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if not username or not password:
            flash('All fields are required!', 'error')
            return render_template('login.html')
        
        # Check user in database
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id, password_hash FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()
        
        if user and check_password_hash(user[1], password):
            session['user_id'] = user[0]
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password!', 'error')
    
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', username=session['username'])

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if 'file' not in request.files:
        flash('No file selected!', 'error')
        return redirect(url_for('dashboard'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected!', 'error')
        return redirect(url_for('dashboard'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Add timestamp to avoid conflicts
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filename = timestamp + filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Predict breed
        predicted_breed, confidence = predict_breed(filepath)
        
        if predicted_breed:
            breed_data = breed_info.get(predicted_breed, {})
            return render_template('result.html', 
                                 breed=predicted_breed,
                                 confidence=confidence,
                                 category=breed_data.get('category', 'Unknown'),
                                 features=breed_data.get('features', []),
                                 image_path=filename)
        else:
            flash('Unable to identify the breed. Please try with a clearer image.', 'error')
            return redirect(url_for('dashboard'))
    else:
        flash('Invalid file type! Please upload PNG, JPG, JPEG, or GIF files.', 'error')
        return redirect(url_for('dashboard'))

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully!', 'success')
    return redirect(url_for('login'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
