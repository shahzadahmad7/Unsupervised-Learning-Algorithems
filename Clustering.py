#Synthesis of Dataset for clustering
from sklearn.datasets import make_classification
from sklearn.cluster import AffinityPropagation
from numpy import where, unique
import matplotlib.pyplot as plt 

#Define Dataset
X,y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)

#Define model
#Value of dampling set lies between 0.5 to 1
model=AffinityPropagation(damping=0.9)

#Other clustering models
#model=AgglomerativeClustering(n_clusters=2)
#model=Birch(threshold=0.01, n_clusters=2)
#model= DBSCAN(eps=0.30, min_samples=9)
#model= OPTICS(eps=0.30, min_samples=9)
#model=KMeans(n_clusters=2)
#model=MiniBatchKMeans(n_clusters=2)
#model=MeanShift()
#model=SpectralClustering(n_clusters=2)
#model=GaussianMixture(n_mixtures=2)

#Fit the model
model.fit(X)

#Assign a cluster to each example
y_hat=model.predict(X)
#Retrieve unique cluster
clusters= unique(y_hat)

#Creating plot for samples from each cluster
for cluster in clusters:
    #Get row indices for samples from each cluster
    row_ix=where(y_hat==cluster)
    #Creating scatter diagram for these samples
    plt.scatter(X[row_ix,0], X[row_ix,1])
plt.show()
