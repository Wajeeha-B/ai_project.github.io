# Contributions - Wajeeha
# K-means clustering:​
# * Elbow method (optimising number of clusters)​   - ​Elbow Method will help find the optimal  number of clusters by identifying the point where increasing clusters doesn’t significantly improve data fit. 
# * Silhouette score    - Silhouette score will be used to assess cluster quality. Higher scores mean better matched data within clusters compared to other clusters. 

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class ElbowMethod:
    def __init__(self, data):
        self.data = data
        self.squared_distances = []
    
    def evaluate(self, max_clusters):
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


from sklearn.metrics import silhouette_score

def calculate_silhouette_score(points, assignment):
    silhouette_avg = silhouette_score(points, assignment)
    return silhouette_avg
