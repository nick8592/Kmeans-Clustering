# Kmeans-Clustering

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

## Precision & Recall

Precision can be seen as a measure of quality, and recall as a measure of quantity. Higher precision means that an algorithm returns more relevant results than irrelevant ones, and high recall means that an algorithm returns most of the relevant results (whether or not irrelevant ones are also returned). [For more...](https://en.wikipedia.org/wiki/Precision_and_recall)   

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
Run K-means Clustering
```
python main.py
```
Run Feature seperately
```
cd features
python <feature>.py <---- replace <feature> with the filename you want to execute
(e.g.) python circle.py
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
