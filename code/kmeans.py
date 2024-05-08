# Contributions - Ashton
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from plotly import express as px, graph_objects as go

class Kmeans:
  def __init__(self, x, y, k, pref) -> None:
    self.x = x
    self.y = y
    self.k = k
    self.pref = pref
    self.rng = np.random.default_rng(10)  # the most random number/ the constructor for creating random numbers

  def cluster(self):
    if self.pref == 'Type':
      mapping = {'h': 0, 'u': 1}

      x_column = self.x[self.pref].copy()

      for key, value in mapping.items():
        x_column.replace(key, value, inplace=True)

      x_column = np.array(x_column)

    else:
      x_column = np.array(self.x[self.pref])
    #   print(x_column)
    

    y_column = np.array(self.y)
    #   print(y_column)

    points = np.hstack((x_column[:, np.newaxis], y_column[:, np.newaxis]))
    
    x_new = self.x
    # x_new['PointsX'] = points[:,0]
    # x_new['PointsY'] = points[:,1]

    # Print the shape of the resulting array
    # print(points.shape)

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
    return x_new, points, centroids


  def plotKmean(self, x, points, centroids, pref):
    plt.scatter(points[:,0], points[:,1], c=x['Assignment' + pref],cmap='Set3')
    plt.scatter(centroids[:,0], centroids[:,1], c='k', marker='*')
    plt.xlabel(pref)  # Set label for x-axis
    plt.ylabel('Price')  # Set label for y-axis
    plt.title('Price Vs ' + self.pref)  # Set title for the plot
    plt.show()