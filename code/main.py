# Contributions - everyone 

import sys
import DataProcessor
from performance_metrics import ElbowMethod
from sklearn.metrics import silhouette_score
sys.path.append('code')


# ----------------parameters------------------
train_size = 0.8 # 80% training data
# columnsToKeep = ['Type', 'Price', 'Distance', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea']

# !!! reducing featues for testing
columnsToKeep = ['Price', 'Size']


filepath = 'dataset/Melbourne_housing_FULL.csv'
prediction_column = 'Price'
# --------------------------------------------

# ---------------Process Data-----------------
dp_obj = DataProcessor.DataProcessor()
dp_obj.LoadData(filepath)

dp_obj.filterData()
dp_obj.keepSelectedColumns(columnsToKeep)
dp_obj.shuffleData()

# !!! reduce data size for testing
dp_obj.reduceDataSize(100)

train_X, train_Y, test_X, test_Y = dp_obj.splitData(train_size, prediction_column)
# --------------------------------------------
# ------------kmeans Clustering---------------
from kmeans import kmeans
k = 5
pref = 'Size'
train_X, points, centroids = kmeans(train_X, train_Y, pref, k)

from matplotlib import pyplot as plt
from plotly import express as px, graph_objects as go

plt.scatter(points[:,0], points[:,1], c=train_X['Assignment'],cmap='Set3')
plt.scatter(centroids[:,0], centroids[:,1], c='k', marker='*')
plt.show()

# ---------------Evaluate using Elbow Method---------------
elbow_method = ElbowMethod(train_X)
elbow_method.evaluate(max_clusters=10)  # Adjust the maximum number of clusters as needed
elbow_method.plot()
print('clustering finished')
# --------------------------------------------

# -------------Compute Silhouette Score---------------
silhouette_avg = silhouette_score(train_X, assignment)
print("Silhouette Score:", silhouette_avg)
# --------------------------------------------
# ---------------Train Model------------------
import NLRegression
nlr = NLRegression.NLRegression(train_X, train_Y, test_X, test_Y, scale_data=False)
nlr.train()
# --------------------------------------------
# -------------Evaluate Model-----------------
nlr.evaluate()

# --------------------------------------------
# -------------Visualize Model----------------
nlr.plot()
# --------------------------------------------

