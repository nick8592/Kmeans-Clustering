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

Precision can be seen as a measure of quality, and recall as a measure of quantity. Higher precision means that an algorithm returns more relevant results than irrelevant ones, and high recall means that an algorithm returns most of the relevant results (whether or not irrelevant ones are also returned). [For more...](https://en.wikipedia.org/wiki/Precision_and_recall)

## Features

### Dominate colors

[Extract dominant colors of an image using Python](https://www.geeksforgeeks.org/extract-dominant-colors-of-an-image-using-python/)  
[ImageDominantColor](https://pypi.org/project/imagedominantcolor/)  
[Finding the Most Common Colors in Python](https://towardsdatascience.com/finding-most-common-colors-in-python-47ea0767a06a)  
[Dominant Color Extraction Dominance and Recoloring](https://github.com/srijannnd/Dominant-Color-Extraction-Dominance-and-Recoloring.git)

### PCA

[PCA Using Python: Image Compression](https://scicoding.com/pca-using-python-image-compression/)

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

> Note: The `1.000` in Precision is incorrect, it may be because of dividing by 0 when calculate the precision, resulting in an output of 1.
