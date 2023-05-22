import cv2
import numpy as np
from scipy.stats import entropy

def calculate_image_entropy(image):
    # Calculate the histogram of pixel intensities
    histogram = np.histogram(image, bins=256, range=(0, 256), density=True)[0]

    # Calculate the entropy using the histogram
    image_entropy = entropy(histogram, base=2)

    return image_entropy

path_1 = "../dataset/train/1_horse/horse_003.jpg"
path_2 = "../dataset/train/2_plane/plane_017.jpg"
path_3 = "../dataset/train/3_rooster/rooster_003.jpg"
path_4 = "../dataset/train/4_tree/tree_007.jpg"
path_5 = "../dataset/train/5_sailboat/sailboat_003.jpg"
path_6 = "../dataset/train/6_motorcycle/motorcycle_003.jpg"
path_7 = "../dataset/train/7_car/car_003.jpg"
path_8 = "../dataset/train/8_butterfly/butterfly_003.jpg"
path_9 = "../dataset/train/9_dragonfly/dragonfly_003.jpg"
path_0 = "../dataset/train/10_flower/flower_004.jpg"

img_path = [path_1, path_2, path_3, path_4, path_5,
            path_6, path_7, path_8, path_9, path_0]
classes = ['horse', 'plane', 'rooster', 'tree', 'sailboat',
           'motorycle', 'car', 'butterfly', 'dragonfly', 'flower']
i = 0
for path in img_path:
    # Load image
    image = cv2.imread(path)

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the entropy of the image
    entropy_value = calculate_image_entropy(image)

    # Print the entropy value
    print(path)
    print(f"Entropy: {entropy_value}")

    cv2.imwrite(f'output/entropy/{classes[i]}.png', gray)

    # Show the final edge image
    cv2.imshow('gray image', gray)

    # Wait for a key press and then close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    i += 1
