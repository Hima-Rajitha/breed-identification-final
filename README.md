# Cattle Breed Classification Web Application

A web application that uses Convolutional Neural Networks (CNN) to identify cattle breeds from uploaded images. The application provides user authentication, image upload functionality, and detailed breed information with features.

## Features

- **User Authentication**: Registration and login system
- **Image Upload**: Drag-and-drop or click-to-upload interface
- **Breed Classification**: CNN-based breed identification
- **Breed Information**: Detailed features and characteristics for each breed
- **Modern UI**: Responsive design with Bootstrap and custom styling

## Supported Cattle Breeds

The application can identify breeds from 5 main categories:

### 1. Cross-bred Dairy Cattle
- Jersey cross

### 2. Exotic Dairy Breeds
- Ayrshire
- Brown Swiss
- Guernsey
- Holstein Friesian
- Jersey
- Red Dane

### 3. Indigenous Dairy Breeds
- Gir
- Red Sindhi
- Sahiwal

### 4. Indigenous Draught Breeds
- Alambadi
- Amritmahal
- Bargur
- Hallikar
- Kangayam
- Khillari
- Pulikulam
- Umblachery

### 5. Indigenous Dual Purpose Breeds
- Deoni
- Hariana
- Kankrej
- Krishna Valley
- Ongole
- Tharparkar

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)

### Steps

1. **Clone or download the project**
   ```bash
   cd breed-classification
   ```

2. **Activate virtual environment** (if using one)
   ```bash
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python run_app.py
   ```

   The script will:
   - Check if a trained model exists
   - Train the model if needed (using your dataset)
   - Start the Flask web server

5. **Access the application**
   - Open your web browser
   - Go to `http://localhost:5000`

## Usage

### 1. Registration
- Click "Register here" on the login page
- Fill in username, email, and password
- Click "Create Account"

### 2. Login
- Enter your username and password
- Click "Login"

### 3. Upload Image
- On the dashboard, click the upload area or drag and drop an image
- Supported formats: PNG, JPG, JPEG, GIF (up to 16MB)
- Click "Identify Breed"

### 4. View Results
- The application will display:
  - The uploaded image
  - Identified breed name
  - Confidence level
  - Breed category
  - 5 key features of the breed

## Project Structure

```
breed-classification/
├── app.py                 # Main Flask application
├── run_app.py            # Application startup script
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── backend/
│   ├── routes/          # API routes (empty - using app.py)
│   └── utils/           # Utility functions (empty - using app.py)
├── frontend/
│   ├── templates/       # HTML templates
│   │   ├── base.html
│   │   ├── register.html
│   │   ├── login.html
│   │   ├── dashboard.html
│   │   └── result.html
│   └── static/          # Static files (CSS, JS, images)
├── model/
│   ├── train_model.py   # Full training script
│   └── create_simple_model.py  # Simple training for small datasets
├── dataset/             # Your cattle breed images
│   ├── Cross_bred-Dairy_Cattle/
│   ├── Exotic dairy breeds of cattle/
│   ├── Indigenous_dairy_breeds_of_cattle/
│   ├── Indigenous_Draught_breeds_of_cattle/
│   └── Indigenous_Dual_purpose_breeds_of_Cattle/
├── uploads/             # Uploaded images (created automatically)
└── venv/               # Virtual environment
```

## Technical Details

### Model Architecture
- **Type**: Convolutional Neural Network (CNN)
- **Input**: 224x224 RGB images
- **Architecture**: 
  - 3 Convolutional blocks with BatchNormalization and Dropout
  - MaxPooling for dimensionality reduction
  - Dense layers for classification
- **Output**: Probability distribution over 25+ cattle breeds

### Technologies Used
- **Backend**: Flask (Python web framework)
- **ML Framework**: TensorFlow/Keras
- **Image Processing**: OpenCV, PIL
- **Database**: SQLite (for user management)
- **Frontend**: HTML5, Bootstrap 5, JavaScript
- **Authentication**: Werkzeug security

### File Handling
- Images are uploaded to the `uploads/` directory
- Automatic filename sanitization and timestamping
- Support for multiple image formats
- 16MB file size limit

## Customization

### Adding New Breeds
1. Add images to the appropriate dataset folder
2. Update the `breed_info` dictionary in `app.py`
3. Retrain the model using `python model/create_simple_model.py`

### Modifying the UI
- Edit templates in `frontend/templates/`
- Customize styles in the `<style>` section of `base.html`
- Add new static files to `frontend/static/`

### Model Improvements
- Modify `model/create_simple_model.py` for architecture changes
- Adjust hyperparameters (learning rate, epochs, batch size)
- Add data augmentation techniques

## Troubleshooting

### Common Issues

1. **Model not found error**
   - Run `python model/create_simple_model.py` manually
   - Check if dataset images are properly organized

2. **Import errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Activate virtual environment if using one

3. **Upload errors**
   - Check file size (must be < 16MB)
   - Ensure file format is supported (PNG, JPG, JPEG, GIF)
   - Verify `uploads/` directory exists and is writable

4. **Low accuracy predictions**
   - Add more training images to each breed folder
   - Retrain the model with more epochs
   - Ensure images are clear and show the cattle clearly

### Performance Tips
- Use clear, well-lit images of cattle
- Ensure the cattle is the main subject in the image
- Avoid heavily cropped or distant images
- Use images with good contrast and resolution

## License

This project is for educational and research purposes. Please ensure you have proper rights to any images used in the dataset.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify your dataset organization matches the expected structure
3. Ensure all dependencies are properly installed
4. Check the console output for error messages


