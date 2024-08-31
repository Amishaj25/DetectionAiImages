
import streamlit as st
import tensorflow as tf
import cv2

# st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('model.hdf5')
    return model
model = load_model()
st.write("""
            # Fake Image Detection
            """
            )

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

import cv2

from PIL import Image, ImageOps
import numpy as np
def import_and_predict(image_data, model):

        size = (32,32)
        image = ImageOps.fit(image_data, size, Image.LANCZOS)
        img = np.asarray(image)
        img_reshape = img[np.newaxis,...]
        prediction = model.predict(img_reshape)
        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    class_names = ['Fake','Real']
    string = "This image most likely is: "+class_names[np.argmax(predictions)]
    st.success(string)

#streamlit run app.py
# Save model architecture to JSON
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

# Save model weights to HDF5
model.save_weights('model_weights.h5')
from tensorflow.keras.models import model_from_json

# Load the model architecture from JSON
with open('model.json', 'r') as json_file:
    model_json = json_file.read()
loaded_model = model_from_json(model_json)

# Load the model weights
loaded_model.load_weights('model_weights.h5')
export_path = "C://Users//amish//Desktop"
import keras
import tensorflow as tf
from tensorflow.keras import models , layers
# Save the model as a SavedModel
tf.saved_model.save(model, export_path)

