from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from datetime import datetime
import os
from PIL import Image
import numpy as np
from predict import LeukemiaClassifier

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'bmp', 'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Initialize classifier
classifier = LeukemiaClassifier('model/model.pth', class_names=['all', 'hem'])

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def is_valid_bmp_blood_cell(image_path):
    """Validate if the image is a proper BMP blood cell image"""
    try:
        with Image.open(image_path) as img:
            # First check if it's a BMP file
            if img.format != 'BMP':
                return False
                
            img = img.convert('RGB')
            
            # Basic size check
            if img.size[0] < 100 or img.size[1] < 100:
                return False
                
            # Color analysis (typical blood cell stains)
            img_array = np.array(img)
            red = img_array[:,:,0].mean()
            blue = img_array[:,:,2].mean()
            
            # Typical blood smear has more blue than red
            if blue / (red + 1e-6) < 0.7:
                return False
                
            return True
            
    except Exception as e:
        app.logger.error(f"Image validation error: {str(e)}")
        return False

@app.route('/')
def home():
    """Render the main index page"""
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Handle image upload and analysis"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded', 'success': False}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file', 'success': False}), 400

    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': 'Invalid file type',
            'details': 'Only image files are supported (BMP, PNG, JPG, JPEG)',
            'suggestion': 'Please upload a valid image file'
        }), 400

    try:
        # Save the uploaded file
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Check if it's a BMP file
        with Image.open(filepath) as img:
            if img.format != 'BMP':
                return jsonify({
                    'success': False,
                    'error': 'Image irrelevnt',
                    'details': 'Please upload a valid blood cell  image',
                    
                    'image_url': f'/uploads/{filename}',
                    'is_bmp': False
                }), 400

        # Validate the image content
        if not is_valid_bmp_blood_cell(filepath):
            return jsonify({
                'success': False,
                'error': 'Invalid blood cell image',
                'details': 'The BMP image does not appear to be a valid blood cell microscopic image',
                'image_url': f'/uploads/{filename}',
                'is_bmp': True
            }), 400
        
        # Get prediction (only reaches here for valid BMP files)
        prediction = classifier.predict(filepath)
        
        if 'error' in prediction:
            return jsonify({
                'success': False,
                'error': prediction['error'],
                'details': 'Error during image analysis',
                'image_url': f'/uploads/{filename}',
                'is_bmp': True
            }), 500
        
        # Prepare response
        response = {
            'success': True,
            'filename': filename,
            'class': prediction['class'],
            'probability': prediction['probability'],
            'all_probability': prediction['all_probabilities'][0],
            'hem_probability': prediction['all_probabilities'][1],
            'is_confident': prediction['is_confident'],
            'image_url': f'/uploads/{filename}',
            'is_bmp': True
        }
        
        if not prediction['is_confident']:
            response['warning'] = 'Low confidence result - please consult a specialist'
        
        return jsonify(response)
        
    except Exception as e:
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({
            'success': False,
            'error': str(e),
            'details': 'An error occurred during analysis'
        }), 500

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True)