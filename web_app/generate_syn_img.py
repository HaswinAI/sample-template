import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the trained generator
generator = load_model('model/gan_generator.h5')

# Generate synthetic images
def generate_synthetic_images(generator, latent_dim, num_images):
    noise = np.random.normal(0, 1, (num_images, latent_dim))
    synthetic_images = generator.predict(noise)
    return synthetic_images

# Generate 1000 synthetic images
synthetic_images = generate_synthetic_images(generator, latent_dim=100, num_images=1000)

# Save synthetic images
synthetic_dir = 'synthetic_data/synthetic_images'
os.makedirs(synthetic_dir, exist_ok=True)
for i, img in enumerate(synthetic_images):
    img = (img * 255).astype(np.uint8)  # Convert to [0, 255] range
    plt.imsave(os.path.join(synthetic_dir, f'synthetic_{i}.png'), img)

print(f"Synthetic images saved to {synthetic_dir}")