import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub  # ✅ Needed for KerasLayer
import numpy as np
import pandas as pd
from PIL import Image

# App title
st.set_page_config(page_title="Dog Vision", layout="centered")
st.title("Dog Vision 🐶")
st.write("Upload a dog image and let the AI predict its breed!")

# Load breed labels (unique, sorted)
def load_breeds(labels_csv):
    df = pd.read_csv(labels_csv)
    breeds = sorted(df['breed'].unique())
    return breeds

# Load model and breeds
@st.cache_resource
def load_model_and_breeds():
    # ✅ Register custom layer
    custom_objects = {'KerasLayer': hub.KerasLayer}
    model = tf.keras.models.load_model(
        "20250725-09491753436958-all-images-mobilenetv2-Adam.h5",
        custom_objects=custom_objects
    )
    breeds = load_breeds("labels.csv")
    return model, breeds

model, breeds = load_model_and_breeds()

# Image preprocessing
def preprocess_image(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img).astype(np.float32)
    arr = arr / 127.5 - 1.0  # MobileNetV2 expects [-1, 1]
    arr = np.expand_dims(arr, axis=0)
    return arr

# Upload image
uploaded_file = st.file_uploader("Choose a dog image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("")
    if st.button("Predict Breed"):
        with st.spinner("Predicting..."):
            arr = preprocess_image(image)
            preds = model.predict(arr)
            top_idx = np.argmax(preds)
            breed = breeds[top_idx]
            confidence = float(preds[0][top_idx])
            st.success(f"**Prediction:** {breed.replace('_', ' ').title()} ({confidence*100:.2f}% confidence)")
            st.bar_chart(pd.Series(preds[0], index=breeds).sort_values(ascending=False)[:5])
else:
    st.info("Please upload an image of a dog to get started.")
