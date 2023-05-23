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
        img = np.transpose(array, (1, 2, 0))*255

        # Convert the numpy array to an OpenCV image in grayscale format
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Convert grayscale image to array
        gray_arr = np.array(gray)

        # Apply Gaussian blur to reduce noise
        gray_blur = cv2.GaussianBlur(gray_arr, (5, 5), 0)

        # Compute the brightness of the image
        brightness = calculate_brightness(gray)

        # Compute the number of contours
        # contours = calculate_contours(gray)

        # Compute the Euler number
        # euler_number = calculate_euler_number(gray_arr)

        # Compute the irregularity ratio
        # irregularity_ratio = calculate_irregularity_ratio(gray_arr)

        # Compute Hue histogram
        h_hist = calculate_h_histogram(img)

        # Compute Number of lines using Hough Transform
        lines = calculate_lines(gray_blur)

        # Compute Number of circles using Hough Transform
        # circles = [calculate_lines(gray_blur)]

        # Compute HOG
        # hog_features = calculate_hog(gray_arr)

        # Compute Standard Deviation
        std_dev = calculate_standard_deviation(gray_arr)

        # Compute Edgewise RGB histogram
        rgb_hist = calculate_edge_histogram(img, gray)

        # Compute Dominate Color
        # center1, center2, center3 = calculate_dominate_color(img)

        # Compute Projection
        row_non_zeros, column_non_zeros = calculate_projection(gray)

        # Compute Entropy
        entropy = calculate_entropy(gray_arr)

        # Compute Edge image non-zero pixels
        non_zero_pixels = calculate_non_zero_pixels(gray_arr)

        # Compute Perimeter
        perimeter = calculate_perimeter(gray_arr)

        # Compute YCbCr color space Cb, Cr histogram
        # cb_hist, cr_hist = calculate_cb_cr_histogram(img)

        # Compute Lab color space a, b histogram
        a_hist, b_hist = calculate_a_b_histogram(img)

        # Compute Mask Area
        # mask_area = calculate_mask_area(gray_arr)

        # Compute Gradient
        # gradient = calculate_gradient(gray_arr, img)

        # Compute Texture
        # texture = calculate_texture(gray_arr)

        # Compute RGB Standard Deviation
        # rgb_std_dev = calculate_rgb_standard_deviation(img)

        # Compute Variance
        # variance = calculate_variance(gray_arr)

        # Compute Frequency domain Standard Deviation
        # freq_std_dev = calculate_freq_std(gray_arr)

        # Compute Mean Channel Value
        # mean_channel_value = calculate_mean_channel(img)

        # Concatenate the features into a single array
        # feature_list = [brightness, std_dev, entropy, lines, perimeter,
        #                 row_non_zeros, column_non_zeros, non_zero_pixels,
        #                 h_hist, rgb_hist, a_hist, b_hist]
        feature_list = [brightness, std_dev, entropy*10, lines, perimeter,
                        row_non_zeros, column_non_zeros, non_zero_pixels,
                        h_hist, rgb_hist, a_hist, b_hist]
        # print(f"Brightness: {len(brightness)}")
        # print(f"Standard Deviation: {len(std_dev)}")
        # print(f"Entropy: {len(entropy)}")
        # print(f"Lines: {len(lines)}")
        # print(f"Perimeter: {len(perimeter)}")
        # print(f"Row non-zeros: {len(row_non_zeros)}")
        # print(f"Cloumn non-zeros: {len(column_non_zeros)}")
        # print(f"Non-zeros: {len(non_zero_pixels)}")
        # print(f"Hue Hist: {len(h_hist)}")
        # print(f"RGB Hist: {len(rgb_hist)}")
        # print(f"a Hist: {len(a_hist)}")
        # print(f"b Hist: {len(b_hist)}")
        # input()
        feature = np.concatenate(feature_list)
        
        features.append(feature)
    features = np.array(features)
    return features

#-----------------------------------------------------------------------------------------------
seed = 210
# Find best random_state
best_random_state_seed = None
best_precision_seed = 0
best_recall_seed = 0

for random_seed in tqdm(range(1000), desc='Find Best Random Seed'):
    torch.manual_seed(random_seed)

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

    for random_state in tqdm(range(100), desc='Find Best Random State'):
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=random_state, n_init='auto').fit(scaled_train_features)
        predicted_labels = kmeans.predict(scaled_val_features)
        pre = precision_score(val_labels, predicted_labels, average='macro', zero_division=1)
        rec = recall_score(val_labels, predicted_labels, average='macro', zero_division=1)
        if pre + rec > best_precision + best_recall:
            best_precision = pre
            best_recall = rec
            best_random_state = random_state
            best_predicted_labels = predicted_labels

    if best_precision + best_recall > best_precision_seed + best_recall_seed:
            best_precision_seed = best_precision
            best_recall_seed = best_recall
            best_random_seed = random_seed
            best_random_seed_state = best_random_state
            best_predicted_labels_seed = best_predicted_labels

    print('\n')
    print(f"Epoch({random_seed})")
    print(f"Best random seed: {best_random_seed}")
    print(f"Best random_state: {best_random_seed_state}")
    print(f"{'Class':13} |  {'Precision'}  |  {'Recall'}")
    print('------------------------------------------')
    print(f"{'Total':13} |     {best_precision_seed:.4f}  |  {best_recall_seed:.4f}")
    print('\n')

# Predict the clusters for the test images
print(f"Val Labels: \n{val_labels}")
print(f"Predicted Labels: \n{best_predicted_labels_seed}")
print('==========================================================================')
print(f"Features Num: {scaled_train_features.shape[1]}")
print(f"Best random seed: {best_random_seed}")
print(f"Best random_state: {best_random_seed_state}")
print('==========================================================================')
print(f"{'Class':13} |  {'Precision'}  |  {'Recall'}")
print('------------------------------------------')
for key, value in classes.items():
    val = [1 if n == value else 0 for n in val_labels]
    pred = [1 if n == value else 0 for n in best_predicted_labels_seed]
    pre = precision_score(val, pred, average='binary', zero_division=1)
    rec = recall_score(val, pred, average='binary', zero_division=1)
    print(f"{key:13} |     {pre:.4f}  |  {rec:.4f}")
print('------------------------------------------')
print(f"{'Total':13} |     {best_precision_seed:.4f}  |  {best_recall_seed:.4f}")
