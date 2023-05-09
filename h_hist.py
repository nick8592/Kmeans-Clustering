from PIL import Image
import numpy as np

# Load the image as a PIL image
# image = Image.open("dataset/train/1_horse/horse_003.jpg")
# image = Image.open("dataset/train/2_plane/plane_002.jpg")
# image = Image.open("dataset/train/3_rooster/rooster_001.jpg")
# image = Image.open("dataset/train/4_tree/tree_001.jpg")
# image = Image.open("dataset/train/5_sailboat/sailboat_001.jpg")
# image = Image.open("dataset/train/6_motorcycle/motorcycle_001.jpg")
# image = Image.open("dataset/train/7_car/car_001.jpg")
# image = Image.open("dataset/train/8_butterfly/butterfly_001.jpg")
# image = Image.open("dataset/train/9_dragonfly/dragonfly_001.jpg")
image = Image.open("dataset/train/10_flower/flower_001.jpg")

img_hsv = image.convert('HSV')
img_array = np.array(img_hsv)
h_channel = img_array[:, :, 0]
hist, bins = np.histogram(h_channel, bins=180, range=[0, 180])
print(type(hist))
print(hist.shape)
print(hist)


import matplotlib.pyplot as plt

plt.bar(bins[:-1], hist, width=1)
plt.xlabel('H value')
plt.ylabel('Frequency')
plt.show()
