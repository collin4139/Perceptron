import os
import numpy as np
import pickle
from PIL import Image
from sklearn.linear_model import Perceptron

# Function to load and preprocess images
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        img = img.resize((28, 28))  # Resize to match the model input size
        images.append(np.array(img).flatten())  # Flatten image to 1D
        labels.append(label)  # 0 for 'L', 1 for 'T'
    return images, labels

# Load images and labels
l_images, l_labels = load_images_from_folder("L_images", 0)
t_images, t_labels = load_images_from_folder("T_images", 1)

# Combine data
X = np.array(l_images + t_images)
y = np.array(l_labels + t_labels)

# Train the perceptron model
model = Perceptron()
model.fit(X, y)

# Save the trained model
with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

print("Model trained and saved as model.pkl")
