import pickle
import numpy as np
import cv2

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

def classify():
    file_input = Element("fileInput").element.files[0]
    reader = window.FileReader.new()
    
    def onload(event):
        img_data = np.frombuffer(event.target.result, dtype=np.uint8)
        img = cv2.imdecode(img_data, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (20, 20)).flatten().reshape(1, -1)
        
        prediction = model.predict(img)
        label = encoder.inverse_transform(prediction)[0]
        
        Element("result").write(f"Predicted Class: {label}")
    
    reader.readAsArrayBuffer(file_input)

