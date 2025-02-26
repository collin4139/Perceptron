import os
import numpy as np
from PIL import Image
import pickle

def load_images(folder, label):
    images, labels = [], []
    
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):  # Ignore non-image files
            continue

        try:
            img = Image.open(img_path).convert("L")  # Convert to grayscale
            img = img.resize((20, 20))  # Resize to 20x20
            img = np.array(img, dtype=np.float32).flatten()  # Flatten the image
            images.append(img)
            labels.append(label)
        except Exception as e:
            print(f"Skipping {img_path}: {e}")  # Ignore corrupted files
    
    return np.array(images), np.array(labels)

# Define paths
L_PATH = "L_images"
T_PATH = "T_images"

# Load datasets
L_images, L_labels = load_images(L_PATH, 0)  # "L" = 0
T_images, T_labels = load_images(T_PATH, 1)  # "T" = 1

# Combine and save
X = np.vstack((L_images, T_images))
y = np.hstack((L_labels, T_labels))

# Save the dataset
with open("dataset.pkl", "wb") as f:
    pickle.dump((X, y), f)

print("✅ Data loading complete. Saved as dataset.pkl")
