# Kmeans-Clustering

## Features
### Dominate colors
[Extract dominant colors of an image using Python](https://www.geeksforgeeks.org/extract-dominant-colors-of-an-image-using-python/)   
[ImageDominantColor](https://pypi.org/project/imagedominantcolor/)   
[Finding the Most Common Colors in Python](https://towardsdatascience.com/finding-most-common-colors-in-python-47ea0767a06a)   
[Dominant Color Extraction Dominance and Recoloring](https://github.com/srijannnd/Dominant-Color-Extraction-Dominance-and-Recoloring.git)

### PCA
[PCA Using Python: Image Compression](https://scicoding.com/pca-using-python-image-compression/)

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
Add features in your `feature_list` at [line 75](https://github.com/nick8592/Kmeans-Clustering/blob/b89ec7397fdb9654cd41f3f98c44e4baa10a77c2/main.py#L75) in `main.py`.   
You can place in any kinds of feature in `feature_list`
For example
```
feature_list = [brightness, euler_number, irregularity_ratio, lines, circles]
```
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
