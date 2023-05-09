from PIL import Image
import numpy as np

def calculate_euler_number(img_arr):
    euler_number = 0
    for i in range(img_arr.shape[0]-1):
        for j in range(img_arr.shape[1]-1):
            a = img_arr[i][j]
            b = img_arr[i][j+1]
            c = img_arr[i+1][j]
            d = img_arr[i+1][j+1]
            if (a != b) or (a != c) or (a != d):
                euler_number += 1
    return euler_number

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
euler_number = calculate_euler_number(img_arr)
print(euler_number)