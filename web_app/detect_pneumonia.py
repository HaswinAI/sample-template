import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import sys
import os

# Load the trained model
model_path = "C:\Haswin info\Medical imaging project\AI_In_Radiography\web_app\model\model.h5"
if not os.path.exists(model_path):
    print("Model not found. Please train the model first.")
    sys.exit(1)

model = load_model(model_path)
print(f"Model loaded from {model_path}.")

# Function to preprocess the input image
def preprocess_image(img_path):
    img = Image.open(img_path)
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    # Check if the image is grayscale (1 channel)
    if img_array.ndim == 2:  # If the image has no color channel (grayscale)
        img_array = np.stack((img_array,)*3, axis=-1)  # Convert to RGB
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to make prediction
def predict_pneumonia(img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    result = 'PNEUMONIA' if prediction > 0.5 else 'NORMAL'
    return result

# Test the model with a sample image
if len(sys.argv) < 2:
    print("Usage: python detect_pneumonia.py <path_to_image>")
    sys.exit(1)

image_path = sys.argv[1]
if not os.path.exists(image_path):
    print(f"Image not found: {image_path}")
    sys.exit(1)

prediction_result = predict_pneumonia(image_path)
print(f"Prediction for {image_path}: {prediction_result}")
