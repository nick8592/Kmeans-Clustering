import numpy as np
from PIL import Image
from skimage import measure
import matplotlib.pyplot as plt

# Load the image as a PIL image
# image = Image.open("dataset/train/1_horse/horse_003.jpg")
# image = Image.open("dataset/train/2_plane/plane_002.jpg")
image = Image.open("../dataset/train/3_rooster/rooster_001.jpg")
# image = Image.open("dataset/train/4_tree/tree_001.jpg")
# image = Image.open("dataset/train/5_sailboat/sailboat_001.jpg")
# image = Image.open("../dataset/train/6_motorcycle/motorcycle_001.jpg")
# image = Image.open("dataset/train/7_car/car_001.jpg")
# image = Image.open("dataset/train/8_butterfly/butterfly_001.jpg")
# image = Image.open("dataset/train/9_dragonfly/dragonfly_001.jpg")
# image = Image.open("dataset/train/10_flower/flower_001.jpg")


# Convert the image to grayscale
gray = image.convert('L')

# Set the threshold value
threshold = 128

# Create a binary image by thresholding the grayscale image
binary = gray.point(lambda x: 0 if x < threshold else 255, '1')

# Convert the grayscale image to a numpy array
img_arr = np.array(binary)

# Find contours in the image
contours = measure.find_contours(img_arr, 0.5)

# Calculate the number of contours
num_contours = len(contours)

print(f"Number of contours: {num_contours}")

# Plot the contours on the original image
fig, ax = plt.subplots()
ax.imshow(image, cmap=plt.cm.gray)

for contour in contours:
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()

# # Find contours in the image with different levels
# levels = np.arange(0.1, 1.0, 0.1)
# fig, axs = plt.subplots(2, 5, figsize=(10, 4))
# for level, ax in zip(levels, axs.ravel()):
#     contours = measure.find_contours(img_arr, level)
#     ax.imshow(img_arr, cmap='gray')
#     for contour in contours:
#         ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
#     ax.set_title(f'Level: {level:.1f}')
#     ax.axis('off')
# plt.show()


