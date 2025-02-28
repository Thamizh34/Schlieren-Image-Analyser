import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image):
    if image is None:
        print("Error: Image not found")
        return None

    # Resize image to standard size (e.g., 500x500)
    image = cv2.resize(image, (500, 500))

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE to enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)

    # Edge Detection using Canny
    edges = cv2.Canny(gray_clahe, 50, 150)

    # Display the processed image using matplotlib
    plt.imshow(edges, cmap='gray')
    plt.title("Edge Detected Image")
    plt.axis("off")
    plt.show()

    return edges

def image_similarity(image1_path, image2_path):
    # Read images
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    if img1 is None or img2 is None:
        print("Error: One or both image paths are invalid.")
        return None

    # Preprocess both images
    img1_edges = preprocess_image(img1)
    img2_edges = preprocess_image(img2)

    if img1_edges is None or img2_edges is None:
        return None

    # Ensure both images have the same dimensions before comparison
    img2_edges = cv2.resize(img2_edges, (img1_edges.shape[1], img1_edges.shape[0]))

    # Compute SSIM
    similarity_index, _ = ssim(img1_edges, img2_edges, full=True)
    print(f"SSIM: {similarity_index:.4f}")

    return similarity_index

# Example Usage
image1 = "F:/C.png"
image2 = "F:/S.png"
similarity = image_similarity(image1, image2)
