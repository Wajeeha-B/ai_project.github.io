# Contributions - Ashton
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from plotly import express as px, graph_objects as go
from scipy.linalg import svd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

class Kmeans:
  def __init__(self, x, y, k, pref) -> None:
    self.x = x
    self.y = y
    self.k = k
    self.pref = pref
    self.rng = np.random.default_rng(10)  # the most random number/ the constructor for creating random numbers

  def cluster(self):
    x_column = np.array(self.x[self.pref])
    # print(x_column)
    
    y_column = np.array(self.y)
    #   print(y_column)

    points = np.hstack((x_column[:, np.newaxis], y_column[:, np.newaxis]))
    # points = np.hstack((x_column, y_column[:, np.newaxis]))
    
    x_new = self.x

    centroids = self.rng.choice(points, size=self.k, replace=False)
    # print("Initial centroids:", centroids)
    assignment = np.zeros(len(points), dtype=int)

    assignment_prev = None

    while assignment_prev is None or any(assignment_prev!=assignment): #ans : assignment_prev is None or any(assignment_prev != assignment)
      assignment_prev= np.copy(assignment) # First keep track of the latest assignment

      for i, point in enumerate(points):
        distances2 = np.sum((centroids - point)**2, axis=1)
        closest_index = np.argmin(distances2)
        assignment[i] = closest_index

      for i in range(self.k):
        centroids[i] = points[assignment==i].mean(axis=0)
      
    # print("Final centroids:", centroids)
    x_new['Assignment' + self.pref] = assignment
    # x_new['Assignment'] = assignment
    return x_new, points, centroids

  def pca(self, prefs):
    # X = np.array(self.x[prefs])
    # B =X-X.mean(axis=0)
    # _, self.s, self.Vh = svd(B, full_matrices=False) # Calculate the SVD on
    # self.coeff = np.dot(np.diag(np.sqrt(self.s)), self.Vh)
    
    X = self.x[prefs]
    X.loc[:, 'Price'] = self.y
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize PCA; specify the number of components to keep
    pca = PCA(n_components=8)

    # Fit PCA on the standardized data
    principal_components = pca.fit_transform(X_scaled)

    # Check the explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_explained_variance = explained_variance.cumsum()

    print("Explained Variance Ratio by Each Component:\n", explained_variance)
    print("Cumulative Explained Variance:\n", cumulative_explained_variance)

    # Choose the number of components that explain up to 95% of the variance
    # n_components = next(i for i, total in enumerate(cumulative_explained_variance) if total > 0.95) + 1
    # print("n_components: ", n_components)

    n_components = 2
    # Apply PCA with the chosen number of components
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X_scaled)

    # Create a DataFrame with the principal components
    pc_columns = [f'PC{i+1}' for i in range(n_components)]
    principal_df = pd.DataFrame(data=principal_components, columns=pc_columns)

    # Example: Add the principal components to the original DataFrame
    df_principal = pd.concat([self.x, principal_df], axis=1)

    # Now you can use df_principal for further analysis or modeling
    print(df_principal.head())
    self.df_principal = df_principal

  def tSNE(self, prefs):

    X = self.x[prefs]
    X.loc[:, 'Price'] = self.y
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize t-SNE to reduce to 2 dimensions
    tsne = TSNE(n_components=2, random_state=42)

    # Fit t-SNE on the standardized data and transform the data
    X_tsne = tsne.fit_transform(X_scaled)

    # Create a DataFrame with the t-SNE components
    tsne_columns = ['tSNE1', 'tSNE2']
    tsne_df = pd.DataFrame(data=X_tsne, columns=tsne_columns)
    self.tsne_df = tsne_df


  def plotKmean(self, x, points, centroids, pref):
    plt.scatter(points[:,0], points[:,1], c=x['Assignment' + pref],cmap='Set3')
    plt.scatter(centroids[:,0], centroids[:,1], c='k', marker='*')
    plt.xlabel(pref)  # Set label for x-axis
    plt.ylabel('Price')  # Set label for y-axis
    plt.title('Price Vs ' + self.pref)  # Set title for the plot
    plt.show()

  def plotPCA(self):
    # plt.scatter(self.coeff[:,0], self.coeff[:,1])
    # plt.quiver(
    #     self.Vh[:,0], self.Vh[:,1],
    #     angles='xy',
    #     scale_units='xy',
    #     scale=1,
    # )
    # plt.axis('equal')
    # plt.show()
    # print("Coefficient:", self.Vh)
    plt.scatter(self.df_principal['PC1'], self.df_principal['PC2'])
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('2D PCA Plot')
    plt.show()

  def plotTSNE(self):
    # Visualize the t-SNE components
    plt.figure(figsize=(8, 6))
    plt.scatter(self.tsne_df['tSNE1'], self.tsne_df['tSNE2'], c='blue', label='Data points')
    plt.title('t-SNE Visualization')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.show()