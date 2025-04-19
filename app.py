import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# Download the model if not already present
@st.cache_resource
def load_model():
    model_path = "autism_model.h5"
    if not os.path.exists(model_path):
        url = "https://drive.google.com/uc?id=1dnmjQRBYE2TAk1JPMlTxJrKaDQlMtq_L"
        gdown.download(url, model_path, quiet=False)
    return tf.keras.models.load_model(model_path)

model = load_model()

# Preprocessing
def preprocess_image(image):
    image = image.resize((256, 256))  # Match the model's expected input size
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image



# UI
st.title("🧠 Autism Detection from Facial Image")
st.write("Upload a facial image to detect signs of autism.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing..."):
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)[0][0]
            label = "Autistic" if prediction > 0.5 else "Non-Autistic"
            confidence = round(prediction * 100, 2) if label == "Autistic" else round((1 - prediction) * 100, 2)
            st.success(f"Prediction: **{label}** ({confidence}% confidence)")
