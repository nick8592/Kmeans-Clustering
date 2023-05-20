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

for path in img_path:
    # Load the image
    image = cv2.imread(path)

    # Get image height, width, channel
    h, w, c = image.shape

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 100, 200)

    # Dilate the edges to connect nearby edges
    kernel = np.ones((5,5), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # Find contours in the dilated edges
    contours, hierarchy = cv2.findContours(dilated_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the index of the largest contour
    largest_contour_index = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))

    # Create a mask for the largest contour and fill it with 
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [contours[largest_contour_index]], 0,  (255, 255, 255), -1)

    # Calculate the number of non-zero pixels
    non_zero_pixels = np.count_nonzero(mask)

    mask_area = round((non_zero_pixels / (h * w)) * 100)

    print(f"Mask Area: {mask_area}%")

    # Show the original image and the segmented image
    cv2.imshow('original', image)
    cv2.imshow('mask', mask)

    # Wait for a key press and then close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()