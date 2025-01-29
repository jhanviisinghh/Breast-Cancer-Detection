from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io
from flask_cors import CORS

# Load both models (normal and masked image models)
normal_model = load_model('trained_model.h5')

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

# Function to preprocess the image
def preprocess_image(image, target_size=(128, 128)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0  # Normalize to [0, 1] range
    return image

# Prediction route for normal images
@app.route('/predict_normal', methods=['POST'])
def predict_normal():
    file = request.files['file']
    if not file:
        return jsonify({'error': 'No file uploaded'})
    
    image = Image.open(io.BytesIO(file.read()))
    processed_image = preprocess_image(image)
    prediction = normal_model.predict(processed_image)
    
    return jsonify({'prediction': prediction.tolist()})

# Prediction route for masked images
@app.route('/predict_masked', methods=['POST'])
def predict_masked():
    file = request.files['file']
    print("Received request...")

    if not file:
        return jsonify({'error': 'No file uploaded'})
    
    image = Image.open(io.BytesIO(file.read()))
    
    processed_image = preprocess_image(image)
    print("Data processed, starting prediction...")

    prediction = normal_model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)  # Get the class with the highest probability
    print(f"Prediction: {prediction}")

    print(f"Predicted class: {predicted_class[0]}")  # Print the predicted class index
    class_names = ['benign', 'malignant',  'normal']
    print(f"Predicted class: {class_names[predicted_class[0]]}")
    return jsonify({
    'prediction': prediction.tolist(),
    'predicted_class': int(predicted_class[0]),
    'class_label': class_names[predicted_class[0]]
})

if __name__ == '__main__':
    app.run(debug=False)
