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
path_0 = "../dataset/train/10_flower/flower_002.jpg"

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

    # Calculate standard deviation
    std_dev = np.std(gray)

    cv2.imwrite(f'output/std_dev/{classes[i]}.png', gray)

    # Show the final edge image
    cv2.imshow('gray image', gray)

    # Output result
    print(path)
    print(f"Standard Deviation: {std_dev}")

    # Wait for a key press and then close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    i += 1
