
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import pairwise_distances

# Compute the Jaccard distance matrix
def jaccard_distance(x, y):
    intersection = np.sum(np.logical_and(x, y))
    union = np.sum(np.logical_or(x, y))
    jaccard_distance= 1 - intersection / union
    distance_matrix = pairwise_distances(X, metric=jaccard_distance)
    return distance_matrix
