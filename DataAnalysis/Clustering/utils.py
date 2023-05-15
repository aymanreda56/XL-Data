import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
#---------------------------------- Data Cleaning Functions ----------------------------------#
def convert_to_numeric(df):
    ''' function to clean numeric columns from any strings and convert them to numeric'''
    # drop nulls
    df.dropna(inplace=True)
    # remove comma from all the columns
    df['Installs'] = df['Installs'].str.replace(',', '').str.replace('+', '').astype(float) 
    df['Rating'] = df['Rating'].astype(float)
    df['Size'] = df['Size'].str.replace(',', '').astype(float) /1000000
    df['Minimum Android'] = df['Minimum Android'].str.replace(',', '').str.replace('Varies with device', '0.0')
    # remove  "and up" from  Minimum Android column
    df['Minimum Android'] = df['Minimum Android'].str.replace(' and up', '')
    return df


# function to convert binary columns to numeric with 0 and 1
def convert_binary(df):
    df.dropna(inplace=True)
    binary_cols = ['Ad Supported', 'In App Purchases', 'Free', 'Editors Choice']
    for col in binary_cols:
        # check if the column is exist in the data frame
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('True', '1')
            df[col] = df[col].astype(str).str.replace('False', '0')
            # drop any value that is not 0 or 1
            df = df[df[col].isin(['0', '1'])]
    return df

def encoding(df,cols, method ='frequency'):
    '''
    Encode the categorical features using the given method
    '''
    if method == 'frequency':
        for col in cols:
            df[col] = df[col].map(df[col].value_counts())/len(df)
    elif method == 'onehot':
        df = pd.get_dummies(df, columns=cols)
    return  df
    # for all columns remove rows with string values
    # for col in df.columns:
    #     if df[col].dtype == 'object':
    #         df = df[df[col].str.isnumeric()]
#---------------------------------- Data Visualization Functions ----------------------------------#
def plot_binary(features_df):
    fig, axes = plt.subplots(2, 2, figsize=(8, 5))
    # loop through the binary columns
    for col, ax in zip(features_df.columns, axes.flatten()):
        # plot a count plot for each column
        sns.countplot(x=col, data=features_df, ax=ax)
        # set the title for each subplot
        ax.set_title(col)
    # set the title for the figure
    fig.suptitle("Binary columns count plots", fontsize=20)
    # set the space between subplots
    fig.tight_layout()
    # show the figure
    plt.show()


#---------------------------------- K-Means Clustering Functions ----------------------------------#
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
   
    return X,kmeans 

def plot_clusters(df, X,kmeans):
    '''
    This function plots the clusters and centroids
    '''
    if X.shape[1] == 3:
        ax = plt.axes(projection ="3d")
        ax.scatter3D(X[:, 0], X[:, 1], X[:, 2],c=kmeans.labels_, cmap='viridis')
        ax.set_xlabel('Size')
        ax.set_ylabel('Installs')
        ax.set_zlabel('Rating')
        plt.show()
    else:
        # Plot the data points and cluster centroids
        plt.scatter(X[:, 0], X[:, 1], X[:, 2],c=kmeans.labels_, cmap='viridis')
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', color='red')
        plt.xlabel(df.columns[0])
        plt.ylabel(df.columns[1])
        plt.show()

def elbow_method(data, max_clusters=10):
    '''
     function to return the optimal number of clusters using the elbow method
      that minimizes the sum of squared distances 
    '''
    # drop all nans 
    data.dropna(inplace=True)
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

#---------------------------------- RDD Clustering Functions ----------------------------------#
def RDD_Kmean(rdd,centroids, max_iter=20):
    
    new_centroids =[]
    # key: index of the mean with min distance
    # value: (Rating, Maximum Installs, Size) 
    i=0
    while i < max_iter:
        i+=1
        
        final_result= rdd.filter(lambda x: x.split(',')[11]!='Varies with device' and x.split(',')[11]!='' and x.split(',')[7]!='' and x.split(',')[3]!='')\
                .map(lambda x: (float(x.split(',')[3]),int(x.split(',')[7]),convert_to_bytes(x.split(',')[11])) )\
                .map(lambda x:(compute_closest_centroid(x[0],x[1],x[2],centroids),(x[0],x[1],x[2])))\
                .mapValues(lambda x: (x[0],x[1],x[2],1,1,1))\
                .reduceByKey(lambda x,y: (x[0]+y[0],x[1]+y[1],x[2]+y[2],x[3]+1,x[4]+1,x[5]+1))\
                .mapValues(lambda x: (round(x[0]/x[3],2),int(x[1]/x[4]),x[2]/x[5]))\
                .mapValues(lambda x: (x[0],x[1],round(x[2]/1000000,2)))
        
        new_centroids= [item[1] for item in np.array(final_result.collect())]
        print("New: ",new_centroids)
        print("Old: ", centroids)
        if centroids != new_centroids :
            centroids = new_centroids
        else:
            break

        return final_result, centroids

def compute_closest_centroid(x, y, z,centroids):
    features = [x, y, z]

    # Calculate the minimum distance between each point and each centroid
    distances = []
    for centroid in centroids:
        distance = 0
        for i in range(len(features)):
            distance += (features[i] - centroid[i]) ** 2
        distances.append(distance ** 0.5)

    # Return the index with the smallest distance
    return distances.index(min(distances))


centroids= [[0.00e+00, 2.21e+02, 1.70e-06],\
             [4.1000e+00, 3.6002e+04, 8.8000e-06],\
             [2.6000e+00, 1.8539e+04, 1.1000e+01]]

def convert_to_bytes(size_str):
    suffixes = {'G': 1000000000, 'M': 1000000, 'k': 1000}
    try:
        for suffix, multiplier in suffixes.items():
            if suffix in size_str:
                return float(size_str[:-1]) * multiplier
    except:
        return 0.0
    

