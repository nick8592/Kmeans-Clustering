import cv2
import numpy as np
from skimage.feature import local_binary_pattern

# Load and preprocess the image
image_path = '../dataset/test/1_horse/horse_01.jpg'

# Load the image
gray_image = cv2.imread(image_path, 0)

# Calculate the Local Binary Pattern (LBP) features
radius = 1
n_points = 8 * radius
lbp_image = local_binary_pattern(gray_image, n_points, radius, method='default')

# Count the number of non-zero pixels in the LBP image
non_zero_pixels = np.count_nonzero(lbp_image)

# Print the number of non-zero pixels
print("Number of non-zero pixels:", non_zero_pixels)

cv2.imshow('lbp image', lbp_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
