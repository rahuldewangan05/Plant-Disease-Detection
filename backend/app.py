from flask import Flask, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# At the start of your app.py
try:
    test_model = tf.keras.models.load_model('models/potato_vgg16.h5', compile=False)
    print("Test model loaded successfully!")
except Exception as e:
    print(f"Test model loading failed: {e}")

# Define plant-specific model paths and classes
PLANT_DATA = {
    'potato': {
        'models': {
            'VGG16': 'models/potato_vgg16.h5',
            'InceptionV3': 'models/potato_inception.h5',
            'DenseNet': 'models/potato_densenet.h5',
            'Bagging': 'models/potato_bagging.h5'
        },
        'classes': ['Early Blight', 'Late Blight','Healthy']
    },
    'tomato': {
        'models': {
            'VGG16': 'models/tomato_vgg16.h5',
            'InceptionV3': 'models/tomato_inception.h5',
            'DenseNet': 'models/tomato_densenet.h5',
            'Bagging': 'models/tomato_bagging.h5'
        },
        'classes': ['Healthy', 'Early Blight', 'Late Blight', 'Septoria Leaf Spot', 'Bacterial Spot']
    },
    'corn': {
        'models': {
            'VGG16': 'models/corn_vgg16.h5',
            'InceptionV3': 'models/corn_inception.h5',
            'DenseNet': 'models/corn_densenet.h5',
            'Bagging': 'models/corn_bagging.h5'
        },
        'classes': ['Healthy', 'Common Rust', 'Gray Leaf Spot', 'Northern Leaf Blight']
    },
    'apple': {
        'models': {
            'VGG16': 'models/apple_vgg16.h5',
            'InceptionV3': 'models/apple_inception.h5',
            'DenseNet': 'models/apple_densenet.h5',
            'Bagging': 'models/apple_bagging.h5'
        },
        'classes': ['Healthy', 'Apple Scab', 'Black Rot', 'Cedar Apple Rust']
    }
}

# Cache for loaded models
MODEL_CACHE = {}

# Image Preprocessing
def preprocess_image(image_path, target_size=(256, 256)):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        # Convert BGR to RGB (OpenCV loads as BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        image = cv2.resize(image, target_size)
        
        # Normalize to [0,1]
        image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    except Exception as e:
        print(f"Error in image preprocessing: {e}")
        return None

def get_model(plant_type, model_name):
    """Get model from cache or load it if not available"""
    model_key = f"{plant_type}_{model_name}"
    
    if model_key not in MODEL_CACHE:
        model_path = PLANT_DATA[plant_type]['models'][model_name]
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            MODEL_CACHE[model_key] = model
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            return None
    
    return MODEL_CACHE[model_key]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        plant_type = request.form.get('plant', '').lower()
        print(f"Processing request for plant type: {plant_type}")
        
        if plant_type not in PLANT_DATA:
            return jsonify({'error': f'Invalid plant type: {plant_type}. Choose from: {list(PLANT_DATA.keys())}'}), 400
        
        # Create temp directory if it doesn't exist
        os.makedirs('temp', exist_ok=True)
        
        # Save image temporarily with unique name
        file_path = f"temp/image_{hash(file.filename)}_{plant_type}.jpg"
        file.save(file_path)
        print(f"Image saved to {file_path}")
        
        # Verify the image was saved correctly
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return jsonify({'error': 'Failed to save image'}), 400
            
        # Preprocess image
        processed_image = preprocess_image(file_path)
        if processed_image is None:
            return jsonify({'error': 'Failed to process image'}), 400
        
        print(f"Image processed successfully, shape: {processed_image.shape}")
        
        # Get predictions from models
        predictions = {}
        classes = PLANT_DATA[plant_type]['classes']
        
        for model_name in PLANT_DATA[plant_type]['models']:
            try:
                model_path = PLANT_DATA[plant_type]['models'][model_name]
                print(f"Loading model: {model_path}")
                
                # Check if model file exists
                if not os.path.exists(model_path):
                    predictions[model_name] = f"Model file not found: {model_path}"
                    continue
                
                model = get_model(plant_type, model_name)
                
                if model is None:
                    predictions[model_name] = "Model unavailable"
                    continue
                    
                print(f"Running prediction with {model_name}")
                pred = model.predict(processed_image)
                pred_class_index = np.argmax(pred, axis=1)[0]
                
                if pred_class_index < len(classes):
                    pred_class = classes[pred_class_index]
                    # Add confidence score
                    confidence = float(pred[0][pred_class_index]) * 100
                    predictions[model_name] = {
                        'disease': pred_class,
                        'confidence': f"{confidence:.2f}%"
                    }
                else:
                    predictions[model_name] = {
                        'disease': 'Unknown',
                        'confidence': 'N/A'
                    }
            except Exception as e:
                print(f"Error in model {model_name}: {str(e)}")
                predictions[model_name] = f"Prediction error: {str(e)}"
        
        # Cleanup
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Failed to remove temp file: {str(e)}")
        
        return jsonify({
            'plant': plant_type,
            'predictions': predictions,
            'available_classes': classes
        })
        
    except Exception as e:
        import traceback
        print(f"Server error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'Server error: {str(e)}'}), 500
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        plant_type = request.form.get('plant', '').lower()
        
        if plant_type not in PLANT_DATA:
            return jsonify({'error': f'Invalid plant type: {plant_type}. Choose from: {list(PLANT_DATA.keys())}'}), 400
        
        # Create temp directory if it doesn't exist
        os.makedirs('temp', exist_ok=True)
        
        # Save image temporarily with unique name
        file_path = f"temp/image_{hash(file.filename)}_{plant_type}.jpg"
        file.save(file_path)
        
        # Preprocess image
        processed_image = preprocess_image(file_path)
        if processed_image is None:
            return jsonify({'error': 'Failed to process image'}), 400
        
        # Get predictions from models
        predictions = {}
        classes = PLANT_DATA[plant_type]['classes']
        
        for model_name in PLANT_DATA[plant_type]['models']:
            model = get_model(plant_type, model_name)
            
            if model is None:
                predictions[model_name] = "Model unavailable"
                continue
                
            try:
                pred = model.predict(processed_image)
                pred_class_index = np.argmax(pred, axis=1)[0]
                
                if pred_class_index < len(classes):
                    pred_class = classes[pred_class_index]
                    # Add confidence score
                    confidence = float(pred[0][pred_class_index]) * 100
                    predictions[model_name] = {
                        'disease': pred_class,
                        'confidence': f"{confidence:.2f}%"
                    }
                else:
                    predictions[model_name] = {
                        'disease': 'Unknown',
                        'confidence': 'N/A'
                    }
            except Exception as e:
                predictions[model_name] = f"Prediction error: {str(e)}"
        
        # Cleanup
        try:
            os.remove(file_path)
        except:
            pass
        
        return jsonify({
            'plant': plant_type,
            'predictions': predictions,
            'available_classes': classes
        })
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)