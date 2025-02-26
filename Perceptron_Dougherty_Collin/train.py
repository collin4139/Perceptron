import os
import numpy as np
import cv2
import pickle
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_data():
    X, y = [], []
    
    for label, folder in enumerate(["L", "T"]):
        path = f"dataset/{folder}"
        for file in os.listdir(path):
            img = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
            img = img.flatten() / 255.0  # Normalize
            X.append(img)
            y.append(label)

    return np.array(X), np.array(y)

# Load and split dataset
X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train perceptron model
model = Perceptron()
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Test accuracy
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
