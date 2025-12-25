# ================================
# Brain Tumor Classification Web App
# Flask + TensorFlow
# ================================

from flask import Flask, render_template, request
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# -------------------------------
# Flask App Initialization
# -------------------------------
app = Flask(__name__)

# Folder to save uploaded images
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create upload folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------------------
# Load Model and Class Names
# -------------------------------
MODEL_PATH = "Brain_Tumors.h5"
CLASS_NAMES_PATH = "class_names.npy"

model = tf.keras.models.load_model(MODEL_PATH)
class_names = np.load(CLASS_NAMES_PATH, allow_pickle=True)

# -------------------------------
# Home Page Route
# -------------------------------
@app.route("/")
def index():
    return render_template("index.html")

# -------------------------------
# Prediction Route
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("index.html", error="No file uploaded")

    file = request.files["file"]

    if file.filename == "":
        return render_template("index.html", error="No selected file")

    # Save uploaded file
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # Image preprocessing
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    confidence = round(float(np.max(prediction) * 100), 2)
    predicted_class = class_names[np.argmax(prediction)]

    return render_template(
        "index.html",
        prediction=predicted_class,
        confidence=confidence,
        image_path=file_path
    )

# -------------------------------
# Run Flask App
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
