import cv2
import numpy as np
from matplotlib import pyplot as plt

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
item = 0
for path in img_path:
    # Load the image
    image = cv2.imread(path)

    # Convert image to Lab color space
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    # Save the ab image
    cv2.imwrite(f'output/Lab/{classes[item]}.png', image_lab)

    # Show the original image and the ab image
    cv2.imshow('Original', image)
    cv2.imshow('Lab image', image_lab)

    # Wait for a key press and then close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    item += 1

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
    # plt.show()
