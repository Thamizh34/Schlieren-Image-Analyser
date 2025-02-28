import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Step 1: Preprocess the schlieren image
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found or unable to load.")
    
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    
    return enhanced

# Step 2: Detect edges
def detect_edges(image):
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    return edges

# Step 4: Detect curved shocks with contours and polynomial fitting
def quadratic_fit(x, a, b, c):
    return a * x**2 + b * x + c

def detect_curved_shocks(edges, original_image):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    curved_shocks = []
    
    for i, contour in enumerate(contours):
        if len(contour) > 50:
            x = contour[:, 0, 0].astype(float)
            y = contour[:, 0, 1].astype(float)
            
            try:
                popt, _ = curve_fit(quadratic_fit, x, y, maxfev=1000)
                a, b, c = popt
                
                x_fit = np.linspace(min(x), max(x), 100)
                y_fit = quadratic_fit(x_fit, a, b, c)
                
                mid_x = (min(x) + max(x)) / 2
                dy_dx = 2 * a * mid_x + b
                angle = np.arctan(dy_dx) * 180 / np.pi
                
                mid_idx = len(x_fit) // 2
                mid_point = (int(x_fit[mid_idx]), int(y_fit[mid_idx]))
                
                label = f"C{i+1}"
                curved_shocks.append({
                    "label": label,
                    "x_fit": x_fit,
                    "y_fit": y_fit,
                    "angle": angle,
                    "midpoint": mid_point
                })
                
                for j in range(len(x_fit) - 1):
                    cv2.line(original_image, 
                            (int(x_fit[j]), int(y_fit[j])), 
                            (int(x_fit[j+1]), int(y_fit[j+1])), 
                            (255, 0, 0), 2)
                cv2.putText(original_image, label, mid_point, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            except RuntimeError:
                continue
    
    return curved_shocks

# Step 5: Analyze and display results
def analyze_schlieren_advanced(image_path):
    preprocessed = preprocess_image(image_path)
    edges = detect_edges(preprocessed)
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    curved_shocks = detect_curved_shocks(edges, original_image)
    

    
    print("\nCurved Shocks (Tangent Angle at Midpoint):")
    for shock in curved_shocks:
        print(f"{shock['label']} (Midpoint: {shock['midpoint']}): {shock['angle']:.2f}°")
    
    plt.figure(figsize=(10, 10))
    
    plt.subplot(2, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.imread(image_path)[..., ::-1])
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.title("Preprocessed CLAHE Image")
    plt.imshow(preprocessed, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.title("Edge Detection")
    plt.imshow(edges, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.title("Detected Shocks")
    plt.imshow(original_image)
    plt.axis('off')
    
    plt.tight_layout(pad=8.0)
    plt.show()

if __name__ == "__main__":
    image_path = "F:/Color_cone_long.png"
    analyze_schlieren_advanced(image_path)
