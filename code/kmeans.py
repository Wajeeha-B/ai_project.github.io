# Contributions - Ashton
import math
import numpy as np
import pandas as pd

rng = np.random.default_rng(4)  # the most random number/ the constructor for creating random numbers

def kmeans(x, y, pref, k):

  x_column = np.array(x[pref])
#   print(x_column)

  y_column = np.array(y)
#   print(y_column)

  points = np.hstack((x_column[:, np.newaxis], y_column[:, np.newaxis]))
  
  x_new = x
  # x_new['PointsX'] = points[:,0]
  # x_new['PointsY'] = points[:,1]

  # Print the shape of the resulting array
  print(points.shape)
  # k = 10

  centroids = rng.choice(points, size=k, replace=False)
  # print("Initial centroids:", centroids)

  assignment = np.zeros(len(points), dtype=int)

  assignment_prev = None

  while assignment_prev is None or any(assignment_prev!=assignment): #ans : assignment_prev is None or any(assignment_prev != assignment)
    assignment_prev= np.copy(assignment) # First keep track of the latest assignment

    for i, point in enumerate(points):
      distances2 = np.sum((centroids - point)**2, axis=1)
      closest_index = np.argmin(distances2)
      assignment[i] = closest_index

    for i in range(k):
      centroids[i] = points[assignment==i].mean(axis=0)
    
  # print("Final centroids:", centroids)
  # print(assignment)
  # print(centroids.shape)
  # print(centroids[:,0])
  # print(centroids[:,1])
  x_new['Assignment'] = assignment
  return x_new, points, centroids