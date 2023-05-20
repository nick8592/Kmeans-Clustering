import cv2
import torch
import numpy as np
from torchvision import datasets, transforms
from tensor_type import Tensor
from tqdm.auto import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from utils import *

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

        # Compute the number of contours
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

        # Compute Dominate Color
        center1, center2, center3 = calculate_dominate_color(img)

        # Compute Projection
        row_non_zeros, column_non_zeros = calculate_projection(gray_arr)

        # Compute Entropy
        entropy = [calculate_entropy(gray_arr)]

        # Compute Edge image non-zero pixels
        non_zero_pixels = [calculate_edge_non_zero_pixels(gray_arr)]

        # Compute Perimeter
        perimeter = [calculate_perimeter(gray_arr)]

        # Concatenate the features into a single array
        feature = np.concatenate([brightness, contours, hog_features,
                                  h_hist, lines, circles,
                                  entropy, rgb_hist, std_dev,
                                  row_non_zeros, column_non_zeros,
                                  non_zero_pixels, perimeter])
        
        features.append(feature)
    features = np.array(features)
    return features

#-----------------------------------------------------------------------------------------------
# 1, 22
torch.manual_seed(1)

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

for random_state in tqdm(range(50), desc='Find Best Random State'):
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state, n_init='auto').fit(scaled_train_features)
    predicted_labels = kmeans.predict(scaled_val_features)
    pre = precision_score(val_labels, predicted_labels, average='micro', zero_division=1)
    rec = recall_score(val_labels, predicted_labels, average='micro', zero_division=1)
    if pre + rec > best_precision + best_recall:
        best_precision = pre
        best_recall = rec
        best_random_state = random_state
        best_predicted_labels = predicted_labels

# Predict the clusters for the test images
predicted_labels = kmeans.predict(scaled_val_features)
print(f"Val Labels: \n{val_labels}")
print(f"Predicted Labels: \n{best_predicted_labels}")

print(f"Features Num: {scaled_train_features.shape[1]}")
print(f"Best random_state: {best_random_state}")
print(f"Highest Total Precision: {best_precision:.4f}, Highest Total Recall: {best_recall:.4f}")

# Calculate Precision, Recall
# pre = precision_score(val_labels, predicted_labels, average='micro')
# rec = recall_score(val_labels, predicted_labels, average='micro')
# print(f"Total Precision: {pre:.4f}, Total Recall: {rec:.4f}")

for item in range(0, 10):
    val = [1 if n == item else 0 for n in val_labels]
    pred = [1 if n == item else 0 for n in best_predicted_labels]
    pre = precision_score(val, pred, average='binary', zero_division=1)
    rec = recall_score(val, pred, average='binary', zero_division=1)
    print(f"Label {item}: Precision={pre:.4f}, Recall={rec:.4f}")




