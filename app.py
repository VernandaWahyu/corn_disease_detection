from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Load the trained model
model = tf.keras.models.load_model('corn_disease_cnn_model.h5', compile=False)


# Define class labels
class_labels = ['Common Rust', 'Gray Leaf Spot', 'Healthy', 'Northern Leaf Blight']

# Create Flask app
app = Flask(__name__)

# Set upload folder
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load and preprocess image
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Predict disease
def predict_disease(img_path):
    img_array = load_and_preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = predictions[0][predicted_class]
    return class_labels[predicted_class], confidence

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Upload and predict route
@app.route('/predict', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            disease, confidence = predict_disease(file_path)
            return render_template('result.html', disease=disease, confidence=confidence, image_path=file_path)

if __name__ == '__main__':
    app.run(debug=True)
