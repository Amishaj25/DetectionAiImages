import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Prevent deprecation warnings
# st.set_option('deprecation.showfileUploaderEncoding', False)

# Cache the model loading for faster performance
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('model.hdf5')
    return model

model = load_model()

# Set the title of the web app
st.write("""
         # Fake Image Detection
         """)

# Allow the user to upload an image
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

# Function to preprocess the image and make predictions
def import_and_predict(image_data, model):
    size = (32, 32)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

# Main logic to handle file upload and predictions
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    class_names = ['Fake', 'Real']
    result = class_names[np.argmax(predictions)]
    st.success(f"This image most likely is: {result}")
