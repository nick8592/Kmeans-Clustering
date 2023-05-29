# Kmeans-Clustering
K-means clustering is a popular unsupervised machine learning algorithm used for grouping similar data points into clusters. It aims to partition a dataset into K clusters, where each data point belongs to the cluster with the nearest mean value.   

To apply K-means clustering to image clustering, you can treat each image as a high-dimensional vector by representing it with features such as color histograms, pixel intensities, or deep learning embeddings.

It's worth noting that K-means clustering can be sensitive to the initialization of cluster centroids. Different initializations may lead to different clustering results. To overcome this, you can run the algorithm multiple times with different initializations and choose the clustering result with the lowest error or highest evaluation metric value.

## Rules
 - Programs must be based on C/C++ or Python w/wo window interface

## Requirements
Please complete the following requirements
1. The dimension of the feature vector must be more than 10.
2. Use K-means clustering to classify the dataset (Total 10 categories)
3. Calculate the **precision rate** and **recall rate** for each category.

## Precision & Recall
Calculate the precision rate and recall rate for each category.
 - Precision $= \space {{True Positive \over Actual Results} \space or \space{True Positive \over True Positive + False Positive}}$
 - Recall $= \space{{True Positive \over Actual Results} \space or \space{True Positive \over True Positive + False Negative}}$

Precision can be seen as a measure of quality, and recall as a measure of quantity. Higher precision means that an algorithm returns more relevant results than irrelevant ones, and high recall means that an algorithm returns most of the relevant results (whether or not irrelevant ones are also returned). [For more...](https://en.wikipedia.org/wiki/Precision_and_recall)


## Features
Here only list out the features that has been used in the `feature_list`.   
Click the arrow to see that feature's advantage.

<details>
  <summary>Brightness</summary>

1. Robustness to Scaling: Brightness is generally invariant to scaling and size changes in the image. This means that even if an image is resized or zoomed in/out, the brightness feature can still provide meaningful information for clustering.
2. Separation of Low and High Contrast Images: Brightness can help differentiate between low and high contrast images. Clustering based on brightness can group together images with similar overall contrast levels, which can be useful for certain applications.
</details>

<details>
  <summary>Standard Deviation</summary>
  
1. Texture Differentiation: Standard deviation is a measure of the variation or spread of pixel values in an image. By using standard deviation as a feature, you can capture information about the texture or fine-grained details present in the image. Images with different textures are likely to have distinct standard deviation values, enabling effective clustering based on texture similarities.
2. Localized Information: Standard deviation can provide insights into localized variations in an image. Areas with high standard deviation are likely to correspond to regions with sharp edges, fine details, or significant texture variations. By clustering images based on standard deviation, you can group together images with similar localized variations or textural patterns.
</details>

<details>
  <summary>Entropy</summary>

1. Texture Analysis: Entropy is particularly useful for capturing texture characteristics in an image. Images with homogeneous or regular textures tend to have low entropy values, indicating low information content, while images with complex or irregular textures have higher entropy values. Clustering based on entropy can help group images with similar textural properties.
2. Scale and Rotation Invariance: Entropy is invariant to scale and rotation transformations. This means that clustering based on entropy can group together images with similar content regardless of their size or orientation. It allows for robust clustering across different scales and orientations.
</details>

<details>
  <summary>Number of Lines</summary>

1. Line Detection: The Hough transform is a powerful technique for detecting lines in an image. By applying the Hough transform, you can identify and extract the lines present in the image. Counting the number of detected lines provides a measure of the line density or complexity, which can be used as a feature for clustering.
2. Structural Information: Lines in an image often represent important structural elements or patterns. By quantifying the number of lines, you capture the structural information of the image. Clustering based on line features allows for grouping images with similar structural characteristics or visual patterns defined by lines.
</details>

<details>
  <summary>Perimeter</summary>

1. Object Size Estimation: The perimeter of the largest contour provides an estimate of the size of the main object in the image. By measuring the length of the contour, you can quantify the object's boundary complexity and approximate its size. Clustering based on the largest contour's perimeter allows for grouping images with similar-sized objects.
2. Shape Information: The largest contour represents the outline or boundary of the main object in the image. Analyzing its perimeter allows you to extract valuable shape information about the object. Characteristics such as curvature, corners, or smoothness can be inferred from the contour's perimeter. Clustering based on these shape features enables grouping images with similar object shapes.
</details>

<details>
  <summary>Hue Histogram</summary>

1. Color-based Representation: The hue histogram captures the distribution of colors in an image, providing a color-based representation that can be effective for clustering tasks. By considering the hue component of the image, you can focus on color information and distinguish images based on their color characteristics.
2. Robustness to Lighting Conditions: The hue component is less sensitive to changes in lighting conditions compared to other color components like saturation or value. This makes the hue histogram more robust to variations in illumination, allowing for clustering based on color similarity across different lighting conditions.
</details>

<details>
  <summary>a, b Channel Histogram (Lab Colorspace)</summary>

1. Discriminative Power: The a and b channel histograms can provide discriminative features that distinguish between images based on their color content. Different color distributions in the histograms indicate varying color characteristics, enabling effective clustering by grouping images with similar color palettes or color distributions.
2. Perceptual Uniformity: The Lab colorspace is designed to be perceptually uniform, meaning that a given distance in the colorspace corresponds to a similar perceptual difference. This property makes the a and b channel histograms effective for capturing human-perceived color differences, enabling meaningful clustering based on color similarity.
</details>

<details>
  <summary>Main Object RGB Histogram</summary>

1. Object-Specific Representation: By focusing on the RGB histogram of the main object in an image, you capture the color distribution specifically related to that object. This allows for clustering based on the color characteristics of the objects, enabling more precise and object-centric grouping.
2. Robustness to Background Variations: By emphasizing the RGB histogram of the main object, you reduce the impact of background variations on the clustering process. The object-specific representation focuses on the colors within the object region, which are less affected by background changes such as lighting conditions or other objects present in the scene.
</details>

<details>
  <summary>Projection (Non-Zero Pixels Along Rows and Columns)</summary>
 
1. Shape Information: The numbers of non-zero pixels along rows and columns provide valuable shape information about the objects in the binary image. By examining the distribution of non-zero pixels in different rows and columns, you can capture the shape characteristics of the objects, such as their elongation, aspect ratio, or symmetry. Clustering based on these shape features allows for grouping images with similar object shapes.
2. Object Localization: Analyzing the distribution of non-zero pixels along rows and columns helps in localizing and extracting the objects from the background. By identifying the rows and columns with higher numbers of non-zero pixels, you can effectively determine the position and extent of the objects. This localization enhances the representation by focusing on the shape-related pixels, reducing the influence of background noise or irrelevant regions.
</details>

<details>
  <summary>Non-Zero Pixels of Binary Image</summary>

1. Object Size Estimation: The number of non-zero pixels in a binary image provides a direct measure of the object size. By counting the non-zero pixels, you can estimate the area or perimeter of the objects. Clustering based on object size allows for grouping images with similar-sized objects.
2. Rotation and Translation Invariance: The numbers of non-zero pixels are invariant to rotation and translation transformations. This means that clustering based on these features can effectively group images with similar object shapes, regardless of their orientation or position within the image. It allows for robust clustering across different object orientations and placements.
</details>

## Installation

Download from github

```
git clone https://github.com/nick8592/Kmeans-Clustering.git
```

Install reqired dependencies

```
pip install -r requirements.txt
```

## Example

Add/Remove features in your `feature_list` in `main.py` at [line 93](https://github.com/nick8592/Kmeans-Clustering/blob/eff03ee50baf5d9ad0eab80e8e692c8bf9757d9e/main.py#L93).  
You can place in any kinds of feature in `feature_list`.

```
(e.g.) feature_list = [brightness, euler_number, irregularity_ratio, lines, circles]
```

Run K-means Clustering

```
python main.py
```

> Real feature method used in `main.py` please check out `utils.py`.

You should see the results looks like below format.

```
Val Labels:
[7 6 0 5 5 7 1 9 8 3 8 5 6 7 9 9 0 2 2 6 4 2 3 4 0 3 7 3 2 0 1 5 9 9 9 7 6
 0 7 9 3 1 6 1 3 2 6 1 1 8 5 5 7 8 5 6 8 5 8 0 8 4 1 3 8 4 3 9 0 4 5 5 6 4
 2 1 1 6 4 4 7 3 0 2 3 8 4 4 0 6 1 9 2 0 2 2 8 7 7 9]
Predicted Labels:
[7 7 3 8 5 7 1 3 3 1 7 1 7 7 0 7 1 2 5 1 7 5 1 1 1 1 7 5 0 5 1 5 1 5 5 3 1
 1 7 1 1 7 8 1 1 1 9 7 1 1 1 8 7 7 5 5 1 5 5 4 5 7 1 7 5 7 1 5 7 1 5 1 7 7
 1 5 7 9 7 7 7 3 4 2 7 1 1 7 3 7 1 8 0 3 8 1 7 7 7 0]
==========================================================================
Features Num: 5
Best random seed: 210
Best random_state: 32
==========================================================================
Class         |  Precision  |  Recall
------------------------------------------
10_flower     |     0.0000  |  0.0000
1_horse       |     0.1935  |  0.6000
2_plane       |     1.0000  |  0.2000
3_rooster     |     0.1429  |  0.1000
4_tree        |     0.0000  |  0.0000
5_sailboat    |     0.2941  |  0.5000
6_motorcycle  |     1.0000  |  0.0000
7_car         |     0.3000  |  0.9000
8_butterfly   |     0.0000  |  0.0000
9_dragonfly   |     0.0000  |  0.0000
------------------------------------------
Total         |     0.2300  |  0.2300
```
Real feature method used in `main.py` please check out `utils.py`

## References
<details>
  <summary>Dominate colors</summary>

[Extract dominant colors of an image using Python](https://www.geeksforgeeks.org/extract-dominant-colors-of-an-image-using-python/)   
[ImageDominantColor](https://pypi.org/project/imagedominantcolor/)   
[Finding the Most Common Colors in Python](https://towardsdatascience.com/finding-most-common-colors-in-python-47ea0767a06a)   
[Dominant Color Extraction Dominance and Recoloring](https://github.com/srijannnd/Dominant-Color-Extraction-Dominance-and-Recoloring.git)
</details>

<details>
  <summary>PCA</summary>

[PCA Using Python: Image Compression](https://scicoding.com/pca-using-python-image-compression/)
</details>
