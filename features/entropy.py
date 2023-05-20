import cv2
import numpy as np
from scipy.stats import entropy

def calculate_image_entropy(image):
    # Calculate the histogram of pixel intensities
    histogram = np.histogram(image, bins=256, range=(0, 256), density=True)[0]

    # Calculate the entropy using the histogram
    image_entropy = entropy(histogram, base=2)

    return image_entropy

# Load and preprocess the image
image_path = '../dataset/test/6_motorcycle/motorcycle_08.jpg'

# Load the image
image = cv2.imread(image_path, 0)

# Calculate the entropy of the image
entropy_value = calculate_image_entropy(image)

# Print the entropy value
print(f"Entropy: {entropy_value}")
