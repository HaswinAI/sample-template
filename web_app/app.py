from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

app = Flask(__name__)

# Path to the model
model_path = os.path.join(app.root_path, 'model', 'model.h5')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")

model = load_model(model_path)

def preprocess_image(img):
    img = img.resize((150, 150))  # Resize to match input size of model
    img = img.convert('RGB')  # Convert to RGB if necessary
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/pneumonia_detection')
def pneumonia_detection():
    return render_template('pneumonia_detection.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        img = Image.open(file)
        processed_img = preprocess_image(img)
        
        # Model prediction
        prediction = model.predict(processed_img)
        prediction = 'PNEUMONIA' if prediction[0][0] > 0.5 else 'NORMAL'
        
        # Render result page with prediction
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
