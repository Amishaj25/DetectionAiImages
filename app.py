import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model.h5')  # Ensure this file exists
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

# Save the model architecture and weights
def save_model(model):
    # Save model architecture to JSON
    model_json = model.to_json()
    with open('model.json', 'w') as json_file:
        json_file.write(model_json)

    # Save model weights to HDF5
    model.save_weights('model_weights.weights.h5')  # Ensure this extension is used

# Save the model
save_model(model)

# Load the model for verification (optional)
def load_saved_model():
    # Load the model architecture from JSON
    with open('model.json', 'r') as json_file:
        model_json = json_file.read()
    
    # Load the model from JSON
    loaded_model = tf.keras.models.model_from_json(model_json)
    
    # Load the model weights
    loaded_model.load_weights('model_weights.weights.hdf5')  # Ensure this extension is used
    
    return loaded_model

loaded_model = load_saved_model()

# Save the model as a SavedModel (optional)
def save_as_saved_model(model, export_path):
    tf.saved_model.save(model, export_path)

# Define the export path
export_path = "C://Users//amish//Desktop/fake_image_detection_model"

# Save the model as a SavedModel
save_as_saved_model(model, export_path)
