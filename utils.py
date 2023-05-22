import cv2
import numpy as np
from skimage import measure
from skimage.feature import hog, graycomatrix, graycoprops
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
    value = [round(value, 4)]
    return value

def calculate_contours(grayscale_image, threshold:int = 128):
    # Create a binary imafrom skimage import measurege by thresholding the grayscale image
    binary = cv2.threshold(grayscale_image, threshold, 255, cv2.THRESH_BINARY)[1]

    # Convert the grayscale image to a numpy array
    img_arr = np.array(binary)

    # Find contours in the image
    contours = measure.find_contours(img_arr, 0.5)

    # Calculate the number of contours
    num_contours = [len(contours)]
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
    return [euler_number]

def calculate_irregularity_ratio(gray_arr):
    std_dev = np.std(gray_arr)
    mean = np.mean(gray_arr)
    irregularity_ratio = [std_dev/mean]
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
    edges = cv2.Canny(cv2.convertScaleAbs(gray_blur), 100, 200)

    # Apply Hough transform to detect lines
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)

    # Output the number of detected lines
    if lines is not None:
        return [len(lines)]
    else:
        return [0]

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
    std_dev = [np.std(gray_arr)]
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
    # Apply Canny edge detection
    edges = cv2.Canny(cv2.convertScaleAbs(gray_arr), 100, 200)

    # Dilate the edges to connect nearby edges
    kernel = np.ones((5,5), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # Find contours in the dilated edges
    contours, hierarchy = cv2.findContours(dilated_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the index of the largest contour
    if len(contours) > 0:
        largest_contour_index = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))

        # Create a mask for the largest contour and fill it with 
        mask = np.zeros_like(gray_arr)
        cv2.drawContours(mask, [contours[largest_contour_index]], 0,  (255, 255, 255), -1)

        # Apply Canny edge detection
        mask_edge = cv2.Canny(cv2.convertScaleAbs(mask), 100, 200)

        # Calculate the non-zero values for each row and column
        row_non_zeros = np.count_nonzero(mask_edge, axis=1).tolist()
        column_non_zeros = np.count_nonzero(mask_edge, axis=0).tolist()
    else:
        row_non_zeros = [0]
        column_non_zeros = [0]

    return row_non_zeros, column_non_zeros

def calculate_entropy(gray_arr):
    # Calculate the histogram of pixel intensities
    histogram = np.histogram(gray_arr, bins=256, range=(0, 256), density=True)[0]

    # Calculate the entropy using the histogram
    entropy_value = [entropy(histogram, base=2)]
    return entropy_value

def calculate_non_zero_pixels(gray_arr):
    # Apply adaptive threshold
    binary = cv2.adaptiveThreshold(cv2.convertScaleAbs(gray_arr), 255, 
                                   cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # Calculate the number of non-zero pixels
    non_zero_pixels = [np.count_nonzero(binary)]
    return non_zero_pixels

def calculate_perimeter(gray_arr):
    # Apply Canny edge detection
    edges = cv2.Canny(cv2.convertScaleAbs(gray_arr), 100, 200)

    # Dilate the edges to connect nearby edges
    kernel = np.ones((5,5), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # Find contours in the dilated edges
    contours, hierarchy = cv2.findContours(dilated_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the index of the largest contour
    if len(contours) > 0:
        largest_contour_index = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))

        # Create a mask for the largest contour and fill it with 
        mask = np.zeros_like(gray_arr)
        cv2.drawContours(mask, [contours[largest_contour_index]], 0,  (255, 255, 255), -1)

        # Apply Canny edge detection
        mask_edge = cv2.Canny(cv2.convertScaleAbs(mask), 100, 200)

        # Calculate the number of non-zero pixels
        perimeter = np.count_nonzero(mask_edge).tolist()
    else:
        perimeter = [0]
    return perimeter

def calculate_cb_cr_histogram(img_rgb):
    # Convert image to YCbCr color space
    image_ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2YCrCb)

    # Split the YCbCr image into channels
    y, cb, cr = cv2.split(image_ycrcb)

    # Calculate histograms for Cb and Cr channels
    hist_cb = cv2.calcHist([cb], [0], None, [256], [0, 256]).flatten()
    hist_cr = cv2.calcHist([cr], [0], None, [256], [0, 256]).flatten()
    return hist_cb, hist_cr

def calculate_a_b_histogram(imag_rgb):
    # Convert image to Lab color space
    image_lab = cv2.cvtColor(imag_rgb, cv2.COLOR_BGR2Lab)

    # Split the Lab image into channels
    l, a, b = cv2.split(image_lab)

    # Calculate histograms for a and b channels
    hist_a = cv2.calcHist([a], [0], None, [256], [0, 256]).flatten()
    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256]).flatten()
    return hist_a, hist_b

def calculate_mask_area(gray_arr):
    # Get image height, width, channel
    h, w = gray_arr.shape

    # Apply Canny edge detection
    edges = cv2.Canny(cv2.convertScaleAbs(gray_arr), 100, 200)

    # Dilate the edges to connect nearby edges
    kernel = np.ones((5,5), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # Find contours in the dilated edges
    contours, hierarchy = cv2.findContours(dilated_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the index of the largest contour
    if len(contours) > 0:
        largest_contour_index = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))

        # Create a mask for the largest contour and fill it with 
        mask = np.zeros_like(gray_arr)
        cv2.drawContours(mask, [contours[largest_contour_index]], 0,  (255, 255, 255), -1)

        # Calculate the number of non-zero pixels
        non_zero_pixels = np.count_nonzero(mask)

        mask_area = [round((non_zero_pixels / (h * w)) * 100)]
    else:
        mask_area = [0]
    return mask_area

def calculate_gradient(gray_arr, image_rgb):
    # 計算灰度影像的梯度
    gradient_x = cv2.Sobel(gray_arr, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray_arr, cv2.CV_64F, 0, 1, ksize=3)

    # 計算每個色彩通道的梯度
    gradient_r = cv2.Sobel(image_rgb[:,:,2], cv2.CV_64F, 1, 1, ksize=3)
    gradient_g = cv2.Sobel(image_rgb[:,:,1], cv2.CV_64F, 1, 1, ksize=3)
    gradient_b = cv2.Sobel(image_rgb[:,:,0], cv2.CV_64F, 1, 1, ksize=3)

    # 計算梯度特徵
    gradient = np.array([np.mean(gradient_x), np.mean(gradient_y),
                         np.mean(gradient_r), np.mean(gradient_g), np.mean(gradient_b)]).tolist()
    return gradient

def calculate_texture(gray_image):
    # 設定GLCM的參數
    distances = [1] # 鄰域距離
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4] # 鄰域方向

    # 計算GLCM
    gray_image = (gray_image * 255).astype(np.uint8)
    glcm = graycomatrix(gray_image, distances=distances, angles=angles, symmetric=True, normed=True)

    # 提取紋理特徵
    # 計算對比度-較高的對比度值表示圖像中不同區域之間的亮度差異較大
    contrast = graycoprops(glcm, prop='contrast')
    # 計算不相似度-較高的不相似度值表示圖像中不同區域之間的灰度差異較大
    dissimilarity = graycoprops(glcm, prop='dissimilarity')
    #計算均勻度-較高的均勻度值表示圖像中不同區域之間的灰度值較為均勻
    homogeneity = graycoprops(glcm, prop='homogeneity')
    # #計算能量-較高的能量值表示圖像中灰度值的分佈較平均
    # energy = graycoprops(glcm, prop='energy')
    #計算相關性-較接近1的相關性值表示圖像中不同區域之間的灰度變化趨勢相似
    correlation = graycoprops(glcm, prop='correlation')
    # #計算角二階矩值-較高的ASM值表示圖像中灰度值的分佈較平坦
    # ASM = graycoprops(glcm, prop='ASM')

    texture_array = np.hstack((contrast, dissimilarity, homogeneity, correlation))
    texture_array = np.squeeze(texture_array)
    texture_list = texture_array.tolist()
    return texture_list

def calculate_rgb_standard_deviation(image_rgb):
    # Calculate the standard deviation of the RGB image
    rgb_std_dev = np.std(image_rgb, axis=(0, 1)).tolist()
    return rgb_std_dev

def calculate_variance(gray_image):
    # Calculate the variance of the grayscale image
    variance = [np.var(gray_image)]
    return variance

def calculate_freq_std(gray_image):
    # Perform Fourier Transform
    f = np.fft.fft2(gray_image)
    fshift = np.fft.fftshift(f)

    # Convert the magnitude spectrum to logarithmic scale
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    # Calculate the standard deviation of the magnitude spectrum
    freq_std_dev = [np.std(magnitude_spectrum)]
    return freq_std_dev

def calculate_mean_channel(image_rgb):
    # Calculate the mean channel value of the RGB image
    mean_channel_value = np.mean(image_rgb, axis=(0, 1)).tolist()
    return mean_channel_value
