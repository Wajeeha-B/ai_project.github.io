# Contributions - Ashton
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from plotly import express as px, graph_objects as go

rng = np.random.default_rng(10)  # the most random number/ the constructor for creating random numbers

class Kmeans:
  def __init__(self, x, y, pref, k) -> None:
    self.x = x
    self.y = y
    self.pref = pref
    self.k = k

  def cluster(self):
    self.x_column = np.array(self.x[self.pref])
    #   print(x_column)

    self.y_column = np.array(self.y)
    #   print(y_column)

    self.points = np.hstack((self.x_column[:, np.newaxis], self.y_column[:, np.newaxis]))
    
    x_new = self.x
    # x_new['PointsX'] = points[:,0]
    # x_new['PointsY'] = points[:,1]

    # Print the shape of the resulting array
    print(self.points.shape)
    # k = 10

    centroids = rng.choice(self.points, size=self.k, replace=False)
    # print("Initial centroids:", centroids)

    assignment = np.zeros(len(self.points), dtype=int)

    assignment_prev = None

    while assignment_prev is None or any(assignment_prev!=assignment): #ans : assignment_prev is None or any(assignment_prev != assignment)
      assignment_prev= np.copy(assignment) # First keep track of the latest assignment

      for i, point in enumerate(self.points):
        distances2 = np.sum((centroids - point)**2, axis=1)
        closest_index = np.argmin(distances2)
        assignment[i] = closest_index

      for i in range(self.k):
        centroids[i] = self.points[assignment==i].mean(axis=0)
      
    # print("Final centroids:", centroids)
    x_new['Assignment'] = assignment
    return x_new, self.points, centroids


  def plotKmean(X, points, centroids):
    plt.scatter(points[:,0], points[:,1], c=X['Assignment'],cmap='Set3')
    plt.scatter(centroids[:,0], centroids[:,1], c='k', marker='*')
    plt.show()