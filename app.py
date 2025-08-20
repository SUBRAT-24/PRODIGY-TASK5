from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Food categories and their average calories (kcal per 100g)
FOOD_CALORIES = {
    'apple': 52,
    'banana': 89,
    'hamburger': 295,
    'pizza': 266,
    'sandwich': 250,
    'salad': 150,
    'french_fries': 312,
    'ice_cream': 207,
    'donut': 452,
    'hot_dog': 290
}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Load the pre-trained model
try:
    model = tf.keras.applications.MobileNetV2(weights='imagenet')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict_food(image_path):
    if model is None:
        return [('error', 0.0, 'Model not loaded')]
        
    try:
        # Preprocess the image
        processed_img = preprocess_image(image_path)
        
        # Make prediction
        predictions = model.predict(processed_img)
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        
        # Filter for food items
        food_predictions = [
            (label, float(prob)) 
            for (_, label, prob) in decoded_predictions 
            if label.lower() in FOOD_CALORIES
        ]
        
        if not food_predictions:
            return [('unknown', 0.0, 'No known food items detected')]
            
        return food_predictions
    except Exception as e:
        print(f"Error in prediction: {e}")
        return [('error', 0.0, str(e))]

def estimate_calories(food_item, probability, portion_size=100):
    """Estimate calories based on food item and portion size (in grams)"""
    food_item = food_item.lower()
    base_calories = FOOD_CALORIES.get(food_item, 0)
    
    # Adjust calories based on portion size (assuming 100g as standard)
    calories = (base_calories * portion_size) / 100
    
    # Adjust confidence score (higher probability = more accurate estimate)
    confidence = min(1.0, probability + 0.2)  # Add small boost to confidence
    
    return {
        'food': food_item.replace('_', ' ').title(),
        'calories': round(calories, 1),
        'confidence': round(confidence * 100, 1),  # as percentage
        'portion_size': portion_size
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get portion size from form or use default (100g)
        try:
            portion_size = float(request.form.get('portion_size', 100))
        except (TypeError, ValueError):
            portion_size = 100
        
        # Make prediction
        predictions = predict_food(filepath)
        
        # Prepare results
        results = []
        for food_item, prob, message in predictions:
            if food_item != 'unknown' and food_item != 'error' and prob > 0.1:  # Only include predictions with >10% confidence
                result = estimate_calories(food_item, prob, portion_size)
                results.append(result)
        
        if not results:
            results = [{
                'food': 'Unknown Food',
                'calories': 0,
                'confidence': 0,
                'portion_size': portion_size,
                'message': 'Could not identify food with high confidence.'
            }]
        
        return jsonify({
            'image_url': f'/uploads/{filename}',
            'results': results
        })
    
    return jsonify({'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('static/uploads', exist_ok=True)
    
    # Run the app
    app.run(debug=True, port=5000)
