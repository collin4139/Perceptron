import os
import numpy as np
import cv2
import random

# Create dataset folders
os.makedirs("dataset/L", exist_ok=True)
os.makedirs("dataset/T", exist_ok=True)

def generate_image(label, size=(28, 28)):
    img = np.ones(size, dtype=np.uint8) * 255  # White background
    thickness = 4

    if label == "L":
        cv2.line(img, (5, 20), (5, 5), (0, 0, 0), thickness)
        cv2.line(img, (5, 20), (20, 20), (0, 0, 0), thickness)
    else:  # "T"
        cv2.line(img, (5, 5), (20, 5), (0, 0, 0), thickness)
        cv2.line(img, (12, 5), (12, 20), (0, 0, 0), thickness)

    return img

# Generate images
for i in range(1000):  # Adjust as needed
    img_L = generate_image("L")
    img_T = generate_image("T")
    
    cv2.imwrite(f"dataset/L/L_{i}.png", img_L)
    cv2.imwrite(f"dataset/T/T_{i}.png", img_T)

print("Dataset created!")
