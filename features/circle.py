import cv2
import numpy as np

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
    # Read the image
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect circles using HoughCircles function
    circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=60, minRadius=0, maxRadius=150)

    # Check if circles were found
    if circles is not None:
        # Convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")

        # Output the number of circles
        print("Number of circles detected:", len(circles))

        # Plot the detected circles on the original image
        for (x, y, r) in circles:
            cv2.circle(img, (x, y), r, (0, 255, 0), 2)

        # Display the image with detected circles
        cv2.imshow("Circles detected", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("No circles detected in the image.")
