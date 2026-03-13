import os
import tensorflow as tf
import numpy as np
from PIL import Image
from detector.views import get_model
from keras.applications.mobilenet_v2 import preprocess_input

def predict_mri(image_path):
    """
    Stand-alone prediction utility matching the web app logic.
    """
    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Get the robustly loaded model
    model = get_model()
    if model is None:
        return "Model Load Error"

    predictions = model.predict(img_array)
    score = predictions[0][0]

    if score >= 0.5:
        return "Alzheimer Detected", score*100
    return "Healthy Brain", (1-score)*100