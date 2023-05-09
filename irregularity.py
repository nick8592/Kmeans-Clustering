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

img_gray = image.convert("L")
img_arr = np.array(img_gray)
std_dev = np.std(img_arr)
mean = np.mean(img_arr)
irregularity_ratio = std_dev/mean
print(irregularity_ratio)


