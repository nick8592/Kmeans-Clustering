# calculate numbers of edge image's non-zero pixel
import cv2
import numpy as np

path_1 = "../dataset/train/1_horse/horse_008.jpg"
path_2 = "../dataset/train/2_plane/plane_001.jpg"
path_3 = "../dataset/train/3_rooster/rooster_001.jpg"
path_4 = "../dataset/train/4_tree/tree_001.jpg"
path_5 = "../dataset/train/5_sailboat/sailboat_001.jpg"
path_6 = "../dataset/train/6_motorcycle/motorcycle_001.jpg"
path_7 = "../dataset/train/7_car/car_001.jpg"
path_8 = "../dataset/train/8_butterfly/butterfly_001.jpg"
path_9 = "../dataset/train/9_dragonfly/dragonfly_001.jpg"
path_0 = "../dataset/train/10_flower/flower_001.jpg"

img_path = [path_1, path_2, path_3, path_4, path_5,
            path_6, path_7, path_8, path_9, path_0]
classes = ['horse', 'plane', 'rooster', 'tree', 'sailboat',
           'motorycle', 'car', 'butterfly', 'dragonfly', 'flower']
i = 0
for path in img_path:
    # Load the image
    image = cv2.imread(path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    threshold1 = 100
    threshold2 = 150
    # edges = cv2.Canny(gray, threshold1, threshold2)

    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # Calculate the number of non-zero pixels
    non_zero_pixels = np.count_nonzero(binary)

    # Print the result
    print(path)
    print("Number of non-zero pixels in the binary image:", non_zero_pixels)

    # Save the final edge image
    cv2.imwrite(f'output/non_zero_pixel/{classes[i]}.png', binary)

    cv2.imshow('binary image', binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    i += 1
