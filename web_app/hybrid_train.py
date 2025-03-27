import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, Concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
from vit_keras import vit  # Vision Transformer library
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Paths to the dataset directories
train_dir = r'C:\Haswin info\Medical imaging project\Xray disease detector\X_ray-dataset\chest_xray\train'
val_dir = r'C:\Haswin info\Medical imaging project\Xray disease detector\X_ray-dataset\chest_xray\val'
test_dir = r'C:\Haswin info\Medical imaging project\Xray disease detector\X_ray-dataset\chest_xray\test'

# Image dimensions
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32

# Image Data Generator (data augmentation)
train_datagen = ImageDataGenerator(rescale=1.0/255.0,
                                   rotation_range=20,
                                   zoom_range=0.15,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.15,
                                   horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Prepare data for training and validation
train_data = train_datagen.flow_from_directory(train_dir,
                                               target_size=(IMG_HEIGHT, IMG_WIDTH),
                                               batch_size=BATCH_SIZE,
                                               class_mode='binary')

val_data = val_datagen.flow_from_directory(val_dir,
                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                           batch_size=BATCH_SIZE,
                                           class_mode='binary')

# Load Vision Transformer (ViT)
vit_model = vit.vit_b16(image_size=IMG_HEIGHT, pretrained=True, include_top=False, pretrained_top=False)

# Freeze ViT layers (optional)
for layer in vit_model.layers:
    layer.trainable = False

# Custom CNN
input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
input_layer = Input(shape=input_shape)

cnn = Conv2D(32, (3, 3), activation='relu')(input_layer)
cnn = MaxPooling2D(2, 2)(cnn)
cnn = BatchNormalization()(cnn)

cnn = Conv2D(64, (3, 3), activation='relu')(cnn)
cnn = MaxPooling2D(2, 2)(cnn)
cnn = BatchNormalization()(cnn)

cnn = Conv2D(128, (3, 3), activation='relu')(cnn)
cnn = MaxPooling2D(2, 2)(cnn)
cnn = Flatten()(cnn)

cnn = Dense(128, activation='relu')(cnn)
cnn = Dropout(0.5)(cnn)

# ViT output
vit_output = vit_model(input_layer)
vit_output = Flatten()(vit_output)

# Feature Fusion
combined = Concatenate()([cnn, vit_output])
output = Dense(1, activation='sigmoid')(combined)  # Binary classification

# Hybrid CNN-ViT Model
model = Model(inputs=input_layer, outputs=output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_data,
                    validation_data=val_data,
                    epochs=10)

# Save the model
os.makedirs("model", exist_ok=True)
model.save('model/hybrid_model.h5')
print("Model saved to model/hybrid_model.h5")

# Load the saved model
model = tf.keras.models.load_model('model/hybrid_model.h5')

# Evaluation
test_data = val_datagen.flow_from_directory(test_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='binary')
test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Visualization
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Grad-CAM for Explainability
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

def grad_cam(model, img_array, layer_name):
    grad_model = Model(inputs=model.inputs, outputs=[model.get_layer(layer_name).output, model.output])
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)[0]
    output = conv_outputs[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))

    cam = np.zeros(output.shape[0:2], dtype=np.float32)
    for index, w in enumerate(weights):
        cam += w * output[:, :, index]

    cam = cv2.resize(cam.numpy(), (IMG_WIDTH, IMG_HEIGHT))
    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    return cam

# Example Grad-CAM usage
img_path = 'path_to_test_image.jpg'  # Replace with actual image path
img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Find the last convolutional layer dynamically
last_conv_layer_name = None
for layer in model.layers[::-1]:  # Iterate in reverse to find the last conv layer
    if isinstance(layer, Conv2D):
        last_conv_layer_name = layer.name
        break

if last_conv_layer_name:
    heatmap = grad_cam(model, img_array, last_conv_layer_name)
    plt.imshow(img_array[0])
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.title('Grad-CAM Heatmap')
    plt.show()
else:
    print("No convolutional layers found for Grad-CAM.")
