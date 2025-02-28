import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image = cv2.imread("F:/S.png")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_image = clahe.apply(gray)

# Optional: Light Gaussian blur to smooth noise
smoothed = cv2.GaussianBlur(clahe_image, (3, 3), 0)

# Display the result
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Grayscale")
plt.imshow(gray, cmap="gray")

plt.subplot(1, 2, 2)
plt.title("CLAHE Enhanced Image")
plt.imshow(smoothed, cmap="gray")

plt.show()

