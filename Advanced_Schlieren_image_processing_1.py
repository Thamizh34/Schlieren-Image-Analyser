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

# Step 3: Detect straight shocks with Hough Transform
def detect_straight_shocks(edges, original_image):
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
    straight_shocks = []
    
    if lines is not None:
        for i, (rho, theta) in enumerate(lines[:, 0]):
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            
            angle = np.abs(theta * 180 / np.pi - 90)  # Angle relative to horizontal
            label = f"S{i+1}"
            straight_shocks.append({
                "label": label,
                "endpoints": ((x1, y1), (x2, y2)),
                "angle": angle,
                "midpoint": (int(x0), int(y0))
            })
            
            # Draw line and label
            cv2.line(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(original_image, label, (int(x0), int(y0)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return straight_shocks

# Step 4: Detect curved shocks with contours and polynomial fitting
def quadratic_fit(x, a, b, c):
    return a * x**2 + b * x + c

def detect_curved_shocks(edges, original_image):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    curved_shocks = []
    
    for i, contour in enumerate(contours):
        if len(contour) > 50:  # Filter small contours
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
                
                # Calculate midpoint for labeling
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
                
                # Draw curve and label
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
    
    # Detect shocks
    straight_shocks = detect_straight_shocks(edges, original_image)
    curved_shocks = detect_curved_shocks(edges, original_image)
    
    # Print results with labels and positions
    print("Straight Shocks:")
    for shock in straight_shocks:
        print(f"{shock['label']} (Midpoint: {shock['midpoint']}): {shock['angle']:.2f}°")
    
    print("\nCurved Shocks (Tangent Angle at Midpoint):")
    for shock in curved_shocks:
        print(f"{shock['label']} (Midpoint: {shock['midpoint']}): {shock['angle']:.2f}°")
    
    # Visualize
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Preprocessed Image")
    plt.imshow(preprocessed, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Edge Detection")
    plt.imshow(edges, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Detected Shocks (Green: Straight, Red: Curved)")
    plt.imshow(original_image)
    plt.axis('off')
    
    plt.show()

# Example usage
if __name__ == "__main__":
    image_path = "F:/Color_cone_long.png"
    analyze_schlieren_advanced(image_path)