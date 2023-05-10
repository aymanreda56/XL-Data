import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def elbow_method(data, max_clusters=10):
    '''
     function to return the optimal number of clusters using the elbow method
      that minimizes the sum of squared distances 
    '''
    # create empty list to store the sum of squared distances
    sse = []
    # create a list of number of clusters
    clusters = range(1, max_clusters)
    # loop through the clusters
    for k in clusters:
        # create a k-means model with k clusters
        kmeans = KMeans(n_clusters=k)
        # fit the model to the data
        kmeans.fit(data)
        # append the sum of squared distances to the list
        sse.append(kmeans.inertia_)
    # plot the elbow curve
    plt.plot(clusters, sse)
    plt.xlabel("Number of clusters")
    plt.ylabel("SSE")
    plt.title("Elbow curve")
    plt.show()
    # return the optimal number of clusters
    return sse.index(min(sse)) + 1



