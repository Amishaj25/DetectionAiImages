import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Set up the Streamlit app
st.set_page_config(page_title="Fake Image Detection", page_icon="📷", layout="centered")

# Load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model.hdf5')  # Ensure this file exists
    return model

model = load_model()

st.title("Fake Image Detection")
st.markdown("""
    ## Welcome to the Fake Image Detection App
    This application can be used to classify images as either 'Fake' or 'Real'.
    Please upload an image in JPG or PNG format, and the model will provide a prediction.
    """)

# File uploader
file = st.file_uploader("Upload an Image", type=["jpg", "png"])

# Function to preprocess the image and make a prediction
def import_and_predict(image_data, model):
    size = (32, 32)  # Adjust the size according to your model's input
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]  # Add batch dimension
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.info("Please upload an image file to get a prediction.")
else:
    st.image(file, caption="Uploaded Image", use_column_width=True)
    st.write("Processing the image...")
    
    # Add a spinner while the prediction is being made
    with st.spinner('Classifying...'):
        predictions = import_and_predict(Image.open(file), model)
        class_names = ['Fake', 'Real']
        predicted_class = class_names[np.argmax(predictions)]
    
    st.success(f"This image is most likely: **{predicted_class}**")

# Add footer or additional information if needed
st.markdown("""
    ---
    Made with ❤️ by Amisha.
    
    """)
