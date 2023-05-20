import cv2
import numpy as np
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt

# Load and preprocess the image
image_path = '../dataset/test/4_tree/tree_07.jpg'

# Load the image
image = cv2.imread(image_path)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply peak_local_max to find local maxima
coordinates = peak_local_max(gray_image, min_distance=10, threshold_abs=50)

# Print the total number of corners detected
num_peaks = len(coordinates)
print("Number of peak local max detected:", num_peaks)

# Plot the result
plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.plot(coordinates[:, 1], coordinates[:, 0], 'r.', markersize=5)
plt.title('Local Maxima')
plt.axis('off')
plt.show()
