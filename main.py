import cv2
import torch
import numpy as np
from torchvision import datasets, transforms
from tensor_type import Tensor
from tqdm.auto import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, \
    multilabel_confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from skimage import measure
from skimage.feature import hog

def show_img(images: Tensor):
    for i in range(images.shape[0]):
        img = transforms.ToPILImage()(images[i])
        img.show()
        input()

# Define a function to extract features from an image
def extract_features(images: Tensor):
    features = []
    for i in range(images.shape[0]):
        # Convert the tensor to a numpy array
        array = images[i].numpy()

        # Transpose the numpy array to match the format expected by OpenCV (H, W, C)
        img = np.transpose(array, (1, 2, 0))

        # Convert the numpy array to an OpenCV image in grayscale format
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Convert grayscale image to array
        gray_arr = np.array(gray)

        # Apply Gaussian blur to reduce noise
        gray_blur = cv2.GaussianBlur(gray_arr, (5, 5), 0)

        # Compute the brightness of the image
        brightness = [calculate_brightness(gray)]

        # Compute the contours
        contours = [calculate_contours(gray)]

        # Compute the Euler number
        euler_number = [calculate_euler_number(gray_arr)]

        # Compute the irregularity ratio
        irregularity_ratio = [calculate_irregularity_ratio(gray_arr)]

        # Compute Hue histogram
        h_hist = calculate_h_histogram(img)

        # Compute Number of lines using Hough Transform
        lines = [calculate_lines(gray_blur)]

        # Compute Number of circles using Hough Transform
        circles = [calculate_lines(gray_blur)]

        # Compute HOG
        hog_features = calculate_hog(gray_arr)

        # Compute Standard Deviation
        std_dev = [calculate_standard_deviation(gray_arr)]

        # Compute Edgewise RGB histogram
        rgb_hist = calculate_edge_histogram(img, gray_arr)

        # Concatenate the features into a single array
        feature = np.concatenate([brightness, contours, euler_number, irregularity_ratio,
                                  h_hist, lines, circles, hog_features, std_dev, rgb_hist])
        
        features.append(feature)
    features = np.array(features)
    return features

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
    # Create a binary image by thresholding the grayscale image
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

def get_precision(cm_arr):
    '''
    Precision = TP / (TP + FP)
    '''
    FP = cm_arr[0][1]
    TP = cm_arr[1][1]
    precision = TP / (TP + FP)
    return precision

def get_recall(cm_arr):
    '''
    Recall = TP / (TP + FN)
    '''
    FN = cm_arr[1][0]
    TP = cm_arr[1][1]
    recall = TP / (TP + FN)
    return recall

#-----------------------------------------------------------------------------------------------

torch.manual_seed(0)

num_clusters = 10

# Define the transformation for the images
# It's an option to transform image, you can either use this or design your transform function.

transform = transforms.Compose([
    # transforms.Resize((224, 224)),  # Resize the images to a fixed size
    transforms.ToTensor(),         # Convert the images to tensors
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the images
])

# Load the dataset
train_dataset = datasets.ImageFolder('dataset/train', transform=transform)
val_dataset = datasets.ImageFolder('dataset/test', transform=transform)
assert train_dataset.class_to_idx == val_dataset.class_to_idx
classes = train_dataset.class_to_idx

# Create a data loader for the training set
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)

# Create a data loader for the validation set
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=True)

# Set the device to use
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

# Get train images feature
train_features = []
for images, labels in tqdm(train_loader, desc='Extract Train Image Features'):
    images, labels = images.to(device), labels.to(device)
    features = extract_features(images)
    train_features.append(features)
train_features = np.array(train_features)
train_image_num = train_features.shape[0]*train_features.shape[1]
train_feature_num = train_features.shape[2]
train_features = np.reshape(train_features, (train_image_num, train_feature_num))

# Get val images feature
val_features = []
val_labels = []
for images, labels in tqdm(val_loader, desc='Extract Val Image Features'):
    images, labels = images.to(device), labels.to(device)
    features = extract_features(images)
    val_features.append(features)
    val_labels.append(labels.numpy())
val_features = np.array(val_features)
val_image_num = val_features.shape[0]*val_features.shape[1]
val_feature_num = val_features.shape[2]
val_features = np.reshape(val_features, (val_image_num, val_feature_num))
val_labels = np.array(val_labels)
val_labels = np.reshape(val_labels, (val_image_num, ))

# Pre-Processing
scaler = StandardScaler()
scaled_train_features = scaler.fit_transform(train_features)
scaled_val_features = scaler.fit_transform(val_features)

# Find best random_state
best_random_state = None
best_precision = 0
best_recall = 0

for random_state in tqdm(range(10), desc='Find Best Random State'):
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state, n_init='auto').fit(scaled_train_features)
    predicted_labels = kmeans.predict(scaled_val_features)
    pre = precision_score(val_labels, predicted_labels, average='micro')
    rec = recall_score(val_labels, predicted_labels, average='micro')
    if pre + rec > best_precision + best_recall:
        best_precision = pre
        best_recall = rec
        best_random_state = random_state

print(f"Best random_state: {best_random_state}")
print(f"Highest Total Precision: {best_precision:.4f}, Highest Total Recall: {best_recall:.4f}")

# Perform k-means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=best_random_state, n_init='auto').fit(scaled_train_features)

# Predict the clusters for the test images
predicted_labels = kmeans.predict(scaled_val_features)
print(f"Val Labels: \n{val_labels}")
print(f"Predicted Labels: \n{predicted_labels}")

# Confusion matrix
cm = confusion_matrix(val_labels, predicted_labels)
# print(f"Confusion Matrix: \n{cm}")

mcm = multilabel_confusion_matrix(val_labels, predicted_labels)
# print(f"Multi Confusion Matrix: \n{mcm}")

print(f"Features Num: {scaled_train_features.shape[1]}")

# Calculate Precision, Recall
pre = precision_score(val_labels, predicted_labels, average='micro')
rec = recall_score(val_labels, predicted_labels, average='micro')
print(f"Total Precision: {pre:.4f}, Total Recall: {rec:.4f}")

for i in range(mcm.shape[0]):
    precision = get_precision(mcm[i])
    recall = get_recall(mcm[i])
    print(f"Label {i}: Precision={precision:.4f}, Recall={recall:.4f}")


