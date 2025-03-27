import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define dataset path
DATASET_PATH = r"C:\Users\Lenovo\OneDrive\Desktop\Alemeno\qr_dataset"

# Load images from a given folder
def load_images(folder, label):
    images, labels = [], []
    for file in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, file), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (128, 128))  # Resize images to 128x128
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

# Load original and counterfeit QR codes
orig_images, orig_labels = load_images(os.path.join(DATASET_PATH, 'first_prints'), 0)
counter_images, counter_labels = load_images(os.path.join(DATASET_PATH, 'second_prints'), 1)

# Combine dataset
X = np.concatenate((orig_images, counter_images), axis=0)
y = np.concatenate((orig_labels, counter_labels), axis=0)
X = X / 255.0  # Normalize pixel values (0-1)

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Flatten for ML models
X_train_flat = X_train.reshape(len(X_train), -1)
X_test_flat = X_test.reshape(len(X_test), -1)

# Train SVM
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_flat, y_train)
print("SVM Accuracy:", accuracy_score(y_test, svm_model.predict(X_test_flat)))

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train_flat, y_train)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_model.predict(X_test_flat)))

# Train KNN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_flat, y_train)
print("KNN Accuracy:", accuracy_score(y_test, knn_model.predict(X_test_flat)))

# Reshape for CNN (Adding 1 channel for grayscale images)
X_train_cnn = X_train.reshape(-1, 128, 128, 1)
X_test_cnn = X_test.reshape(-1, 128, 128, 1)

# Build CNN Model
cnn_model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile CNN
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train CNN & Store Training History
history = cnn_model.fit(X_train_cnn, y_train, epochs=100, validation_data=(X_test_cnn, y_test))

# Evaluate CNN
y_pred_cnn = (cnn_model.predict(X_test_cnn) > 0.5).astype("int32")
print("CNN Accuracy:", accuracy_score(y_test, y_pred_cnn))

# Print Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_cnn))

# Save trained CNN model
MODEL_PATH = "qr_auth_model.h5"
cnn_model.save(MODEL_PATH)
print(f"Model saved successfully at {MODEL_PATH}")

# -------------------------------------------
#  ADDING GRAPH TO CHECK OVERFITTING/UNDERFITTING
# -------------------------------------------

# Plot Accuracy Graph
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Plot Loss Graph
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)
plt.show()
