import cv2
import numpy as np

def detect_sailboat(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define the lower and upper bounds for the sailboat color in HSV
    lower_color = np.array([100, 100, 100])
    upper_color = np.array([130, 255, 255])
    
    # Threshold the image to extract the sailboat color region
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    # Perform morphological operations to remove noise and refine the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    sailboat_detected = False
    
    # Iterate over the contours and check if they represent a sailboat
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Adjust this threshold based on your image
            # Draw a bounding rectangle around the contour
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            sailboat_detected = True
    
    # Show the original image with bounding boxes (if any)
    cv2.imshow('Sailboat Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return sailboat_detected

# Specify the path to your image
image_path = '../dataset/test/5_sailboat/sailboat_09.jpg'

# Detect sailboat in the image
sailboat_detected = detect_sailboat(image_path)

if sailboat_detected:
    print("Sailboat detected!")
else:
    print("No sailboat detected.")


