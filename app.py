"""Streamlit UI: upload an image, get a generated caption.

Run with:  streamlit run app.py
"""
import streamlit as st
from PIL import Image

from src.predict import get_predictor

st.set_page_config(page_title="Image Caption Generator", page_icon="🖼️")
st.title("🖼️ Image Caption Generator")
st.write("Upload an image and the model will generate a natural-language caption for it.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Generating caption..."):
        predictor = get_predictor()
        caption = predictor.predict(image)

    st.success(f"**Caption:** {caption}")
