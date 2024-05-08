# Contributions - Wajeeha
# K-means clustering:​
# * Elbow method (optimising number of clusters)​   - ​Elbow Method will help find the optimal number of clusters by identifying the point where increasing clusters doesn’t significantly improve data fit. 
# * Silhouette score    - Silhouette score will be used to assess cluster quality. Higher scores mean better matched data within clusters compared to other clusters. 

import matplotlib.pyplot as plt
import numpy as np
import Kmeans


class ElbowMethod:
    def __init__(self, kmeans_instance):
        self.kmeans_instance = kmeans_instance
        self.inertia_values = []
    
    def evaluate(self, max_clusters):
        for k in range(1, max_clusters + 1):
            self.kmeans_instance.k = k
            x_new, points, centroids = self.kmeans_instance.cluster()  # Cluster for the current k
            inertia = np.sum((points - centroids)**2)  # Calculate the inertia
            self.inertia_values.append(inertia)
    
    def plot(self):
        plt.plot(range(1, len(self.inertia_values) + 1), self.inertia_values, marker='o')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal K')
        plt.show()



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


