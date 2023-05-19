import cv2
import numpy as np
from skimage import morphology, feature

# Skeletonization
def skeletonize_image(image):
    binary_image = image > 0  # Binarize the image
    skeleton = morphology.skeletonize(binary_image)
    return skeleton

# Feature Extraction
def extract_skeleton_features(skeleton):
    # Extract features from the skeleton (e.g., branch points, endpoints)
    branch_points = feature.branch_point_detection(skeleton)
    endpoints = feature.endpoint_detection(skeleton)
    # Additional feature extraction steps can be added here
    features = np.concatenate((branch_points, endpoints), axis=0)
    return features

# Load and preprocess the image
image_path = '../dataset/test/4_tree/tree_02.jpg'

# Load the image
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding (adjust the threshold value as needed)
edges = cv2.Canny(cv2.convertScaleAbs(gray), 100, 150, apertureSize=3)
cv2.imshow('Edge', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Find contours in the thresholded image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour
largest_contour = max(contours, key=cv2.contourArea)

# Draw the largest contour on the original image
image_with_contour = cv2.drawContours(image.copy(), [largest_contour], -1, (0, 255, 0), 2)

# Display the image with the largest contour
cv2.imshow('Largest Contour', image_with_contour)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Skeletonization
skeletonized_image = skeletonize_image(image_with_contour)

cv2.imshow('Skeleton Detection', skeletonized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
