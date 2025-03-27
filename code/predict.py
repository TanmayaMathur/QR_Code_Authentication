import cv2
import numpy as np
import tensorflow as tf
from config import MODEL_PATH, IMG_SIZE

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

def predict_qr(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0  # Normalize
    img = img.reshape(1, 128, 128, 1)

    prediction = model.predict(img)[0][0]
    if prediction > 0.5:
        return "Counterfeit QR Code"
    else:
        return "Original QR Code"

# Example usage
image_path = "WhatsApp Image 2025-03-27 at 12.15.07_8096cbfd.jpg"
print(predict_qr(image_path))
