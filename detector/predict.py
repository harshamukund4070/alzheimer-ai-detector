import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("detector/models/alzheimer_binary_model.h5")

def predict_mri(image_path):

    img = Image.open(image_path).convert("RGB")   # 🔥 important fix
    img = img.resize((224,224))

    img = np.array(img)/255.0
    img = np.expand_dims(img,axis=0)

    pred = model.predict(img)[0][0]

    if pred > 0.5:
        return "Healthy", pred*100
    else:
        return "Alzheimer Detected", (1-pred)*100