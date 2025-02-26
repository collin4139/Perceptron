import os
from PIL import Image
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define dataset path
L_PATH = "L_images/"
T_PATH = "T_images/"

# Load images and labels
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (20, 20)).flatten()  # Resize to 20x20 and flatten
            images.append(img)
            labels.append(label)
    return images, labels

# Load both classes
L_images, L_labels = load_images_from_folder(L_PATH, "L")
T_images, T_labels = load_images_from_folder(T_PATH, "T")

# Combine dataset
X = np.array(L_images + T_images)
y = np.array(L_labels + T_labels)

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)  # Convert "L" and "T" to numeric labels

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Perceptron model
model = Perceptron(max_iter=1000, tol=1e-3)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save model and encoder
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("Training complete. Model and encoder saved.")
