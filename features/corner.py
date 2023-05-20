import cv2
import numpy as np
from skimage.feature import corner_harris, corner_peaks
import matplotlib.pyplot as plt

# Load and preprocess the image
image_path = '../dataset/test/5_sailboat/sailboat_03.jpg'

# Load the image
image = cv2.imread(image_path)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply the Harris corner detection
corner_response = corner_harris(gray_image)

# Find the local maxima in the corner response image
corner_coords = corner_peaks(corner_response, min_distance=3)

# Print the total number of corners detected
num_corners = len(corner_coords)
print("Number of corners detected:", num_corners)

# Plot the corners on the original image
plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.plot(corner_coords[:, 1], corner_coords[:, 0], 'r.', markersize=5)
plt.title('Corners Detected')
plt.axis('off')
plt.show()
