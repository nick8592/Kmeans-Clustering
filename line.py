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

img_path = path_0
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Apply Gaussian blur with a kernel size of 5x5
blurred = cv2.GaussianBlur(img, (5, 5), 0)

# Apply edge detection (can be skipped if the input image is already an edge map)
edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

# Apply Hough transform to detect lines
lines = cv2.HoughLines(edges, 1, np.pi/180, 100)

# Check if any lines were detected
if lines is None:
    print(f"Lines Num: 0")
else:
    # Output the number of detected lines
    print(f"Lines Num: {len(lines)}")

    # Draw lines on the original image
    color_img = cv2.imread(img_path)
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(color_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Display the result
    cv2.imshow("Detected Lines", color_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
