import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("model/model.h5")

def predict_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("❌ Image not found. Check file path.")

    # Convert grayscale to RGB if needed
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.reshape(img, (1, 128, 128, 3))

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        return f"PNEUMONIA ({prediction*100:.2f}%)"
    else:
        return f"NORMAL ({(1-prediction)*100:.2f}%)"