import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# Define the test directory path
test_dir = r'C:\Haswin info\Medical imaging project\AI_In_Radiography\X_ray-dataset\chest_xray\test'

# Define the ImageDataGenerator for testing (No augmentation for test data)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Load the saved model
model = tf.keras.models.load_model('model\model.h5', compile=False)

# Compile the model before evaluation
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Create test data generator
test_data = test_datagen.flow_from_directory(
    test_dir, 
    target_size=(150, 150), 
    batch_size=32, 
    class_mode='binary',
    shuffle=False
)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_data, verbose=1)
print(f"✅ Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"✅ Test Loss: {test_loss:.4f}")

# Generate predictions
predictions = model.predict(test_data)
predicted_classes = (predictions > 0.5).astype(int).flatten()

# Get true labels
true_labels = test_data.classes

# ================== 1️⃣ Confusion Matrix ==================
cm = confusion_matrix(true_labels, predicted_classes)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# ================== 2️⃣ Classification Report ==================
print("\n📊 Classification Report:\n", classification_report(true_labels, predicted_classes, target_names=['Normal', 'Pneumonia']))

# ================== 3️⃣ ROC Curve ==================
fpr, tpr, _ = roc_curve(true_labels, predictions)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random chance line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()
