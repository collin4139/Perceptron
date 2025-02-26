import pickle
import numpy as np
import cv2
from pyodide.http import open_url

# Load trained model
model_url = open_url("https://your-github.io/model.pkl")
model = pickle.load(model_url)

def preprocess_image(img_file):
    img = cv2.imdecode(np.frombuffer(img_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28)).flatten() / 255.0
    return np.array([img])

def predict():
    file_input = document.getElementById("upload")
    file = file_input.files[0]

    if file:
        img_data = preprocess_image(file)
        prediction = model.predict(img_data)[0]
        result_text = "L" if prediction == 0 else "T"
        document.getElementById("result").innerText = f"Prediction: {result_text}"
