import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import skimage
import numpy as np

path = '../dataset/train/2_plane/plane_007.jpg'

image_orig = skimage.io.imread(path) # Load image

width = image_orig.shape[0]
height = image_orig.shape[1]
channels = image_orig.shape[2]

image = np.reshape(image_orig, (width, height*channels))

# Utility function that compresses image with given number
# of principal components
def compress_image(n_components, image, size):
    pca = PCA(n_components=n_components)
    image_compressed = pca.fit_transform(image)
    print(pca.components_.shape)
    return pca.inverse_transform(image_compressed).reshape(size).astype('uint8')

# Compress images with different numbers of principal components
image_6 = compress_image(6, image, image_orig.shape)
image_36 = compress_image(36, image, image_orig.shape)
image_95 = compress_image(95, image, image_orig.shape)
image_283 = compress_image(283, image, image_orig.shape)
# image_583 = compress_image(583, image, image_orig.shape)

fig, axes = plt.subplots(2,3, figsize=(10,5), constrained_layout=True)

axes[0][0].imshow(image_6)
axes[0][0].set_title("Compressed image (PCA: 6)")

axes[0][1].imshow(image_36)
axes[0][1].set_title("Compressed image (PCA: 36)")

axes[0][2].imshow(image_95)
axes[0][2].set_title("Compressed image (PCA: 95)")

axes[1][0].imshow(image_283)
axes[1][0].set_title("Compressed image (PCA: 283)")

# axes[1][1].imshow(image_583)
# axes[1][1].set_title("Compressed image (PCA: 583)")

axes[1][2].imshow(image_orig)
axes[1][2].set_title("Original image")
plt.show()

