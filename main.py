import torch
from torchvision import datasets, transforms
from torchvision import transforms
import numpy as np
from PIL import Image, ImageFilter
from tensor_type import Tensor
from tqdm.auto import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, \
    multilabel_confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

def show_img(images: Tensor, labels: Tensor):
    for i in range(images.shape[0]):
        img = transforms.ToPILImage()(images[i])
        print(labels[i].item())
        img.show()
        input()

# Define a function to extract features from an image
def extract_features(images: Tensor):
    features = []
    for i in range(images.shape[0]):
        img = transforms.ToPILImage()(images[i])
        # Convert the image to grayscale
        gray = img.convert('L')

        # Compute the edges
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edges = transforms.ToTensor()(edges)

        # Compute the RGB color histogram
        r, g, b = img.split()
        r_hist = r.histogram()
        g_hist = g.histogram()
        b_hist = b.histogram()

        # Concatenate the features into a single array
        feature = np.concatenate([edges.flatten(),
                                  r_hist, g_hist, b_hist])
        features.append(feature)
    features = np.array(features)
    return features

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

# Create a data loader for the training set
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)

# Create a data loader for the validation set
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=True)

# Set the device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
print(f"Val Labels: \n{val_labels}")

# Pre-Processing
scaler = StandardScaler()
scaled_train_features = scaler.fit_transform(train_features)
scaled_val_features = scaler.fit_transform(val_features)

# Perform k-means clustering
centroids = scaled_train_features[np.random.choice(scaled_train_features.shape[0], num_clusters, replace=False)]
for i in tqdm(range(100), desc='K-means'):
    distances = np.sqrt(((scaled_train_features - centroids[:, np.newaxis])**2).sum(axis=2))
    labels = np.argmin(distances, axis=0)
    for j in range(num_clusters):
        centroids[j] = scaled_train_features[labels == j].mean(axis=0)

# Predict the clusters for the test images
distances = np.sqrt(((scaled_val_features - centroids[:, np.newaxis])**2).sum(axis=2))
predicted_labels = np.argmin(distances, axis=0)
print(f"Predicted Labels: \n{predicted_labels}")


# # sklearn k-meanse
# kmeans = KMeans(init="random",n_clusters=num_clusters,n_init=10,max_iter=300,random_state=42)
# fit = kmeans.fit(scaled_train_features)
# inertia = kmeans.inertia_
# center = kmeans.cluster_centers_
# itr = kmeans.n_iter_
# print(fit)
# print(inertia)
# print(center.shape)
# print(itr)
# input()


# Confusion matrix
cm = confusion_matrix(val_labels, predicted_labels)
print(f"Confusion Matrix: \n{cm}")

mcm = multilabel_confusion_matrix(val_labels, predicted_labels)
print(f"Multi Confusion Matrix: \n{mcm}")

# Calculate Precision, Recall
pre = precision_score(val_labels, predicted_labels, average='micro')
rec = recall_score(val_labels, predicted_labels, average='micro')
print(f"Total Precision: {pre}")
print(f"Total Recall:    {rec}")

for i in range(mcm.shape[0]):
    precision = get_precision(mcm[i])
    recall = get_recall(mcm[i])
    print(f"{i}. Precision: {precision}")
    print(f"{i}. Recall:    {recall}")


