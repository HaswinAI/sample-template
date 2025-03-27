from flask import Flask, request, render_template
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from model import load_model, predict_image

app = Flask(__name__)

# Load model once when the Flask app starts
model = load_model()

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)

    # Perform prediction
    prediction, confidence = predict_image(model, filepath)

    return render_template('result.html', prediction=prediction, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
