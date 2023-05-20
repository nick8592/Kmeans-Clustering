# calculate numbers of edge image's non-zero pixel
import cv2
import numpy as np

# Load and preprocess the image
image_path = '../dataset/test/3_rooster/rooster_04.jpg'

# Load the image
image = cv2.imread(image_path, 0)

# Apply edge detection
threshold1 = 100
threshold2 = 150
edges = cv2.Canny(image, threshold1, threshold2)

# Calculate the number of non-zero pixels
non_zero_pixels = np.count_nonzero(edges)

# Print the result
print("Number of non-zero pixels in the edge image:", non_zero_pixels)

cv2.imshow('edge image', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
