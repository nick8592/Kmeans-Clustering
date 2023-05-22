import cv2
import numpy as np

path_1 = "../dataset/train/1_horse/horse_003.jpg"
path_2 = "../dataset/train/2_plane/plane_002.jpg"
path_3 = "../dataset/train/3_rooster/rooster_001.jpg"
path_4 = "../dataset/train/4_tree/tree_003.jpg"
path_5 = "../dataset/train/5_sailboat/sailboat_001.jpg"
path_6 = "../dataset/train/6_motorcycle/motorcycle_001.jpg"
path_7 = "../dataset/train/7_car/car_001.jpg"
path_8 = "../dataset/train/8_butterfly/butterfly_001.jpg"
path_9 = "../dataset/train/9_dragonfly/dragonfly_001.jpg"
path_0 = "../dataset/train/10_flower/flower_001.jpg"

img_path = path_4
# Load the image
image = cv2.imread(img_path)

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# calculate gray image's gradient
gradient_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
gradient_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

# calculate rgb image's gradient
gradient_r = cv2.Sobel(image[:,:,2], cv2.CV_64F, 1, 1, ksize=3)
gradient_g = cv2.Sobel(image[:,:,1], cv2.CV_64F, 1, 1, ksize=3)
gradient_b = cv2.Sobel(image[:,:,0], cv2.CV_64F, 1, 1, ksize=3)

# combine gradient features
gradient = np.array([np.mean(gradient_x), np.mean(gradient_y),
np.mean(gradient_r), np.mean(gradient_g), np.mean(gradient_b)])

print(gradient)