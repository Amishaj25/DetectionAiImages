import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model.hdf5')  # Ensure this file exists
    return model

model = load_model()

st.write("""
    # Fake Image Detection
""")

# File uploader
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

# Function to preprocess the image and make a prediction
def import_and_predict(image_data, model):
    size = (32, 32)  # Adjust the size according to your model's input
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]  # Add batch dimension
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    class_names = ['Fake', 'Real']
    predicted_class = class_names[np.argmax(predictions)]
    st.success(f"This image most likely is: {predicted_class}")


