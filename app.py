import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Page setup
st.set_page_config(page_title="AI Medical Image Analysis", layout="centered")

# Load model (cache for performance)
@st.cache_resource
def load_my_model():
    return load_model("model/model.h5")

model = load_my_model()

# Title
st.markdown("<h1 style='text-align: center;'>🩺 AI-Powered Medical Image Analysis</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Detect Pneumonia from Chest X-ray</h4>", unsafe_allow_html=True)

st.write("Upload a chest X-ray image and the AI model will predict whether it is **Normal** or **Pneumonia**.")

# Upload
uploaded_file = st.file_uploader("📤 Upload X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray Image", width=250)

    st.markdown("### 🔍 Processing...")

    # Convert image
    img = np.array(image)

    # Fix grayscale
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Resize + Normalize
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img, verbose=0)[0][0]

    st.markdown("### 🧠 Prediction Result")

    # Output
    if prediction > 0.5:
        confidence = prediction * 100
        st.error("🦠 Pneumonia Detected")
    else:
        confidence = (1 - prediction) * 100
        st.success("✅ Normal")

    # Confidence bar
    st.progress(int(confidence))
    st.write(f"Confidence: **{confidence:.2f}%**")

    # Divider
    st.markdown("---")

    # Footer
    st.warning("⚠️ This AI system is for educational purposes only and should not replace professional medical diagnosis.")