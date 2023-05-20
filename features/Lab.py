import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load and preprocess the image
image_path = '../dataset/test/1_horse/horse_02.jpg'

# Load the image
image = cv2.imread(image_path)

# Convert image to Lab color space
image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

# Split the Lab image into channels
l, a, b = cv2.split(image_lab)

# Calculate histograms for a and b channels
hist_a = cv2.calcHist([a], [0], None, [256], [0, 256]).flatten()
hist_b = cv2.calcHist([b], [0], None, [256], [0, 256]).flatten()

# Display the histograms
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(hist_a, color='g')
plt.title('a Histogram')
plt.xlabel('Bins')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.plot(hist_b, color='r')
plt.title('b Histogram')
plt.xlabel('Bins')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
