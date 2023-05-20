import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load and preprocess the image
image_path = '../dataset/test/1_horse/horse_02.jpg'

# Load the image
image = cv2.imread(image_path)

# Convert image to YCbCr color space
image_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

# Split the YCbCr image into channels
y, cb, cr = cv2.split(image_ycrcb)

# Calculate histograms for Cb and Cr channels
hist_cb = cv2.calcHist([cb], [0], None, [256], [0, 256]).flatten()
hist_cr = cv2.calcHist([cr], [0], None, [256], [0, 256]).flatten()

# Display the histograms
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(hist_cb, color='b')
plt.title('Cb Histogram')
plt.xlabel('Bins')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.plot(hist_cr, color='r')
plt.title('Cr Histogram')
plt.xlabel('Bins')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
