# Import library
from clustimage import Clustimage

# init
cl = Clustimage(method='pca')

# Note that you manually need to download the data from the caltech website and simply provide the directory where the images are stored.
# Download dataset

# Cluster  images in path location.
results = cl.fit_transform('.//101_ObjectCategories//', min_clust=60, max_clust=110)

# If you want to experiment with a different clustering and/or evaluation approach, use the cluster functionality.
# This will avoid pre-processing, and performing the feature extraction of all images again.
# You can also cluster on the 2-D embedded space by setting the cluster_space parameter 'low'
#
# cluster(cluster='agglomerative', evaluate='dbindex', metric='euclidean', linkage='ward', min_clust=15, max_clust=200, cluster_space='high')

# Cluster evaluation plots such as the Silhouette plot
cl.clusteval.plot()
cl.clusteval.scatter(cl.results['xycoord'])

# PCA explained variance plot
cl.pca.plot()

# Dendrogram
cl.dendrogram()

# Plot unique image per cluster
cl.plot_unique(img_mean=False)

# Scatterplot
cl.scatter(dotsize=8, zoom=0.2, img_mean=False)

# Plot images per cluster or all clusters
cl.plot(labels=8)