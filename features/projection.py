import cv2
import numpy as np

# Read the binary image
image_path = '../dataset/train/9_dragonfly/dragonfly_002.jpg'
image = cv2.imread(image_path, 0)

# Convert grayscale image to array
gray_arr = np.array(image)
edges = cv2.Canny(cv2.convertScaleAbs(gray_arr), 50, 150, apertureSize=3)

# Calculate the non-zero values for each row and column
row_non_zeros = np.count_nonzero(edges, axis=1)
column_non_zeros = np.count_nonzero(edges, axis=0)

# Print the lists
print("Row non-zero amounts:")
print(row_non_zeros)
print("\nColumn non-zero amounts:")
print(column_non_zeros)

cv2.imshow('Edge', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()