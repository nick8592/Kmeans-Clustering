import cv2
import numpy as np
from skimage import measure
from skimage.feature import hog
from sklearn.cluster import KMeans
from scipy.stats import entropy

def calculate_brightness(grayscale_image):
    # Calculate histogram
    hist, bins = np.histogram(grayscale_image.ravel(), 256, [0, 256])

    pixels = sum(hist)
    brightness = scale = len(hist)

    for index in range(0, scale):
        ratio = hist[index] / pixels
        brightness += ratio * (-scale + index)

    value = 1 if brightness == 255 else brightness / scale
    value = round(value, 4)
    return value

def calculate_contours(grayscale_image, threshold:int = 128):
    # Create a binary imafrom skimage import measurege by thresholding the grayscale image
    binary = cv2.threshold(grayscale_image, threshold, 255, cv2.THRESH_BINARY)[1]

    # Convert the grayscale image to a numpy array
    img_arr = np.array(binary)

    # Find contours in the image
    contours = measure.find_contours(img_arr, 0.5)

    # Calculate the number of contours
    num_contours = len(contours)
    return num_contours

def calculate_euler_number(gray_arr):
    euler_number = 0
    for i in range(gray_arr.shape[0]-1):
        for j in range(gray_arr.shape[1]-1):
            a = gray_arr[i][j]
            b = gray_arr[i][j+1]
            c = gray_arr[i+1][j]
            d = gray_arr[i+1][j+1]
            if (a != b) or (a != c) or (a != d):
                euler_number += 1
    return euler_number

def calculate_irregularity_ratio(gray_arr):
    std_dev = np.std(gray_arr)
    mean = np.mean(gray_arr)
    irregularity_ratio = std_dev/mean
    return irregularity_ratio

def calculate_h_histogram(img_rgb):
    # Convert RGB image to HSV color space
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # Extract the H channel from HSV image
    h_channel = img_hsv[:, :, 0]
    
    # Calculate the histogram
    hist, bins = np.histogram(h_channel, bins=180, range=[0, 180])
    return hist

def calculate_lines(gray_blur):
    # Apply edge detection (can be skipped if the input image is already an edge map)
    edges = cv2.Canny(cv2.convertScaleAbs(gray_blur), 50, 150, apertureSize=3)

    # Apply Hough transform to detect lines
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)

    # Output the number of detected lines
    if lines is not None:
        return len(lines)
    else:
        return 0

def calculate_circles(gray_blur):
    # Detect circles using HoughCircles function
    circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=60, minRadius=0, maxRadius=150)

    if circles is not None:
        return len(circles)
    else:
        return 0

def calculate_hog(gray_arr):
    # Perform HOG feature extraction
    hog_features = hog(gray_arr, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    return hog_features

def calculate_standard_deviation(gray_arr):
    # Calculate standard deviation
    std_dev = np.std(gray_arr)
    return std_dev

def calculate_edge_histogram(img_rgb, gray_arr):
    # Apply Canny edge detection
    edges = cv2.Canny(cv2.convertScaleAbs(gray_arr), 100, 200)

    # Dilate the edges to connect nearby edges
    kernel = np.ones((5,5), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # Find contours in the dilated edges
    contours, hierarchy = cv2.findContours(dilated_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Check that contours is not empty
    if len(contours) > 0:
        # Find the index of the largest contour
        largest_contour_index = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))

        # Find the index of the largest contour
        largest_contour_index = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))

        # Create a mask for the largest contour and fill it with 
        mask = np.zeros_like(gray_arr)
        cv2.drawContours(mask, [contours[largest_contour_index]], 0,  (255, 255, 255), -1)

        # Calculate the area of the mask
        mask_area = cv2.countNonZero(mask)

        # Calculate the RGB histogram of the masked image
        masked_img = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
        color = ('b','g','r')
        histograms = []
        for i, col in enumerate(color):
            hist = cv2.calcHist([masked_img], [i], mask, [256], [0, 256])
            histograms.append(hist)
    else:
        # Handle the case where no contours were found
        largest_contour_index = None
        color = ('b','g','r')
        histograms = []
        for i, col in enumerate(color):
            hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
            histograms.append(hist)
    r_hist = np.asarray(histograms[0]).reshape((256,))
    g_hist = np.asarray(histograms[1]).reshape((256,))
    b_hist = np.asarray(histograms[2]).reshape((256,))
    rgb_hist_array = np.concatenate([r_hist, g_hist, b_hist])
    return rgb_hist_array

def calculate_dominate_color(img_rgb):
    clt = KMeans(n_clusters=3, n_init='auto')
    centers = clt.fit(img_rgb.reshape(-1, 3))
    center1 = (centers.cluster_centers_[0]*255).astype(int)
    center2 = (centers.cluster_centers_[1]*255).astype(int)
    center3 = (centers.cluster_centers_[2]*255).astype(int)
    return center1, center2, center3

def calculate_projection(gray_arr):
    # Apply edge detection (can be skipped if the input image is already an edge map)
    edges = cv2.Canny(cv2.convertScaleAbs(gray_arr), 100, 150)
    # Calculate the non-zero values for each row and column
    row_non_zeros = np.count_nonzero(edges, axis=1)
    column_non_zeros = np.count_nonzero(edges, axis=0)
    return row_non_zeros, column_non_zeros

def calculate_entropy(gray_arr):
    # Calculate the histogram of pixel intensities
    histogram = np.histogram(gray_arr, bins=256, range=(0, 256), density=True)[0]

    # Calculate the entropy using the histogram
    entropy_value = entropy(histogram, base=2)
    return entropy_value

def calculate_edge_non_zero_pixels(gray_arr):
    # Apply edge detection
    threshold1 = 100
    threshold2 = 150
    edges = cv2.Canny(cv2.convertScaleAbs(gray_arr), threshold1, threshold2)

    # Calculate the number of non-zero pixels
    non_zero_pixels = np.count_nonzero(edges)
    return non_zero_pixels
