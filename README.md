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
Add features in your `feature_list` in `main.py` at [line 75](https://github.com/nick8592/Kmeans-Clustering/blob/b89ec7397fdb9654cd41f3f98c44e4baa10a77c2/main.py#L75).   
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
[9 7 3 3 5 2 1 1 6 2 7 6 3 0 4 8 1 6 1 1 6 9 3 7 4 5 7 5 2 6 9 7 5 3 4 0 2
 5 4 8 8 5 0 9 9 4 2 3 5 6 6 1 3 0 2 8 8 0 2 9 3 6 9 4 8 9 8 4 1 7 0 5 5 4
 9 1 0 2 7 8 7 3 6 5 0 2 4 8 8 9 0 0 3 7 6 1 1 7 4 2]
Predicted Labels: 
[3 7 3 3 8 8 3 3 7 7 7 2 3 5 3 3 8 7 3 3 8 9 3 7 3 8 7 7 9 2 9 7 8 3 7 3 3
 7 3 3 3 8 3 9 9 3 8 3 8 7 7 7 7 3 9 3 3 3 9 3 3 7 3 3 3 9 3 3 3 3 3 8 7 3
 9 3 3 8 7 3 7 3 7 7 3 8 3 3 3 9 7 5 3 7 9 3 3 7 7 8]
Features Num: 64397
Best random_state: 23
Highest Total Precision: 0.2500, Highest Total Recall: 0.2500
Label 0: Precision=1.0000, Recall=0.0000
Label 1: Precision=1.0000, Recall=0.0000
Label 2: Precision=0.0000, Recall=0.0000
Label 3: Precision=0.1915, Recall=0.9000
Label 4: Precision=1.0000, Recall=0.0000
Label 5: Precision=0.0000, Recall=0.0000
Label 6: Precision=1.0000, Recall=0.0000
Label 7: Precision=0.3600, Recall=0.9000
Label 8: Precision=0.0000, Recall=0.0000
Label 9: Precision=0.6364, Recall=0.7000
```
> Note: The `1.000` in Precision is incorrect, it may be because of dividing by 0 when calculate the precision, resulting in an output of 1.   