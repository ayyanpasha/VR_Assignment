import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Updated detect_coins function in coin_detection.py
def detect_coins(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return None, None, 0
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 2)  # Increased blur kernel
    
    # Tuned HoughCircles parameters
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2, 
        minDist=70,        # Increased minimum distance between coins
        param1=50, 
        param2=45,         # Stricter accumulator threshold
        minRadius=30,      # Adjusted based on actual coin sizes
        maxRadius=80
    )
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # Remove duplicate circles (e.g., concentric detections)
        filtered_circles = []
        for (x, y, r) in circles:
            overlap = False
            for (x2, y2, r2) in filtered_circles:
                distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                if distance < max(r, r2):
                    overlap = True
                    break
            if not overlap:
                filtered_circles.append((x, y, r))
        count = len(filtered_circles)
        
        # Draw and segment using filtered circles
        for (x, y, r) in filtered_circles:
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
        cv2.imwrite("detected_coins.jpg", output)
        
        # ... rest of segmentation code ...
        
        return output, filtered_circles, count
    else:
        print("No coins detected.")
        return None, None, 0
if __name__ == "__main__":
    image_path = "coins.jpg"
    result, _, count = detect_coins(image_path)
    if result is not None:
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title(f"Detected Coins: {count}")
        plt.axis('off')
        plt.show()
        print(f"Total coins: {count}")