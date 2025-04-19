import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("autism_model.h5")

model = load_model()

# Preprocessing function
def preprocess_image(img):
    img = img.resize((128, 128))  # Change if your model expects different size
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Streamlit UI
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
            st.success(f"Prediction: **{label}**")

