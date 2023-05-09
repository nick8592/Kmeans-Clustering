import cv2
from skimage.feature import hog

path_1 = "dataset/train/1_horse/horse_003.jpg"
path_2 = "dataset/train/2_plane/plane_002.jpg"
path_3 = "dataset/train/3_rooster/rooster_001.jpg"
path_4 = "dataset/train/4_tree/tree_001.jpg"
path_5 = "dataset/train/5_sailboat/sailboat_001.jpg"
path_6 = "dataset/train/6_motorcycle/motorcycle_001.jpg"
path_7 = "dataset/train/7_car/car_001.jpg"
path_8 = "dataset/train/8_butterfly/butterfly_001.jpg"
path_9 = "dataset/train/9_dragonfly/dragonfly_001.jpg"
path_0 = "dataset/train/10_flower/flower_001.jpg"

img_path = [path_1, path_2, path_3, path_4, path_5,
            path_6, path_7, path_8, path_9, path_0]

for path in img_path:
    # Load image
    img = cv2.imread(path)

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Perform HOG feature extraction
    hog_features = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))

    # Print the HOG feature vector
    print(hog_features.shape)
    print(hog_features)