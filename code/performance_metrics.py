# Contributions - Wajeeha
# K-means clustering:​
# * Elbow method (optimising number of clusters)​   - ​Elbow Method will help find the optimal number of clusters by identifying the point where increasing clusters doesn’t significantly improve data fit. 
# * Silhouette score    - Silhouette score will be used to assess cluster quality. Higher scores mean better matched data within clusters compared to other clusters. 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class ElbowMethod:
    def __init__(self, data):
        self.data = data
        self.squared_distances = []
        self.optimal_k = None
    
    def evaluate(self, max_clusters=10):
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(self.data)
            self.squared_distances.append(kmeans.inertia_)
    
    def plot(self):
        plt.plot(range(1, len(self.squared_distances) + 1), self.squared_distances, marker='o')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Sum of Squared Distances')
        plt.title('Elbow Method for Optimal K')
        plt.show()
    
    def optimal_number_of_clusters(self):
        # Find the elbow point (where the rate of decrease slows down)
        deltas = np.diff(self.squared_distances, 2)
        curvature = np.abs(deltas)
        self.optimal_k = np.argmax(curvature) + 1
        return self.optimal_k


def calculate_silhouette_score(points, assignment):
    silhouette_avg = 0
    n = len(points)
    
    # Calculate the silhouette score for each sample
    for i in range(n):
        # Calculate the average distance from the current point to points in the same cluster (a)
        a = np.mean([np.linalg.norm(points[i] - points[j]) for j in range(n) if assignment[j] == assignment[i] and j != i])
        
        # Calculate the average distance from the current point to points in the nearest other cluster (b)
        b = min([np.mean([np.linalg.norm(points[i] - points[j]) for j in range(n) if assignment[j] == c and c != assignment[i]]) for c in set(assignment) if c != assignment[i]])
        
        # Calculate the silhouette score for the current point
        silhouette = (b - a) / max(a, b)
        
        # Accumulate the silhouette score for all points
        silhouette_avg += silhouette
    
    # Compute the average silhouette score across all samples
    silhouette_avg /= n
    
    return silhouette_avg


