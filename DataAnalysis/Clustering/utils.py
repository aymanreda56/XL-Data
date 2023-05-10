import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

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
    return clusters.index(min(sse)) + 1


def silhouette_method(data, max_clusters=10):
    '''
    function to return the optimal number of clusters according
     to the silhouette so that silhouette score is maximized
    '''
    # create empty list to store the silhouette scores
    silhouette_scores = []
    # create a list of number of clusters
    clusters = range(2, max_clusters)
    # loop through the clusters
    for k in clusters:
        # create a k-means model with k clusters
        kmeans = KMeans(n_clusters=k)
        # fit the model to the data
        kmeans.fit(data)
        # append the silhouette score to the list
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))
    # plot the silhouette curve
    plt.plot(clusters, silhouette_scores)
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette curve")
    plt.show()
    # return the optimal number of clusters
    return clusters.index(max(silhouette_scores)) + 2

def convert_to_numeric(df):
    ''' function to clean numeric columns from any strings and convert them to numeric'''
    # drop nulls
    df.dropna(inplace=True)
    # remove comma from all the columns
    df['Installs'] = df['Installs'].str.replace(',', '').str.replace('+', '').astype(float) 
    df['Maximum Installs'] = df['Maximum Installs'].str.replace(',', '').astype(float)
    df['Minimum Installs'] = df['Minimum Installs'].str.replace(',', '').astype(float)
    df['Rating'] = df['Rating'].astype(float)
    df['Size'] = df['Size'].str.replace(',', '').astype(float) /1000000
    df['Minimum Android'] = df['Minimum Android'].str.replace(',', '').str.replace('Varies with device', '0.0')
    # remove  "and up" from  Minimum Android column
    df['Minimum Android'] = df['Minimum Android'].str.replace(' and up', '')
    return df

def euclidean_distance(x1, x2):
    '''
    function to calculate the euclidean distance between two points
    '''
    return np.sqrt(np.sum((x1 - x2)**2))

def kmeans_plus_plus(df, K):
    '''
    function return initial centroids for data to be used in 
    k-mean clustering using k-mean++ approach 
    '''
    #convert data frame to numpy array X
    X = df.to_numpy()
    # Initialize first centroid randomly
    means = [X[np.random.choice(len(X))]]
    
    for k in range(1, K):
        # Calculate distance between each data point and the nearest centroid
        distances = np.array([min([np.linalg.norm(x-c)**2 for c in means]) for x in X])
        
        # Calculate probability of each data point being chosen
        probs = distances / distances.sum()
        
        # Randomly select a data point as the next centroid
        next_mean = X[np.random.choice(len(X), p=probs)]
        
        # Add the next centroid to the list of centroids
        means.append(next_mean)

    print("Initial centroids: ", means)
    return means

def k_mean(df , k=3, initial_centroids=None):

    '''
    function to perform k-mean clustering on the data, plot the clusters and return them
    '''
     
    #convert data frame to numpy array X
    X = df.to_numpy()

    # Perform k-means clustering with k clusters
    kmeans = KMeans(n_clusters=k, init=initial_centroids)
    kmeans.fit(X)

    # Plot the data points and cluster centroids
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', color='red')
    plt.show()

    # Get the clusters as lists
    cluster_0 = X[kmeans.labels_ == 0].tolist()
    cluster_1 = X[kmeans.labels_ == 1].tolist()
    cluster_2 = X[kmeans.labels_ == 2].tolist()

    #return the clusters
    return cluster_0, cluster_1, cluster_2

