# Import relevant files and libraries
import DataProcessor
import sys
import LRegression
import NLRegression
import performance_metrics
import Kmeans
import UserInterface
import numpy as np
import pandas as pd

# Allocate 80% of data to training
train_size = 0.7

# Identify features to be used
columnsToKeep = ['Price','Type','Bedroom','Bathroom','Car','Landsize', 'BuildingArea', 'Latitude','Longitude']

# Path to dataset.
filepath = './dataset/Melbourne_housing_FULL.csv'

# Identify prediction (ground truth)
prediction_column = 'Price'


# Create a DataProcessor object and load the data
dp_obj = DataProcessor.DataProcessor()
dp_obj.LoadData(filepath)

# Display the data
print(dp_obj.getData())

# Remove incomplete points, remove unused features and shuffle the data
dp_obj.keepSelectedColumns(columnsToKeep)
dp_obj.filterMelbourneData()

dp_obj.remove_outliers(columnsToKeep, plot=False)

# dp_obj.encodeCategoricalData(['Type'])
dp_obj.shuffleData()
dp_obj.reduceDataSize(1000) # remove this to train on the full dataset


columnsToCorrelate = ['Price','Bedroom','Bathroom','Car','Landsize', 'BuildingArea', 'Latitude','Longitude']
dp_obj.calculateCorelation(columnsToCorrelate)

dp_obj.plotDataNormalDistribution()

# Split the data into training and testing
train_X, train_Y, test_X, test_Y = dp_obj.splitData(train_size, prediction_column)

# print(train_X.head())

# Remove dwelling type and additional landsize / building area data for some processing
clean_columns = ['Type', 'Landsize']
train_X_clean = train_X.drop(columns=clean_columns,axis=1)
test_X_clean = test_X.drop(columns=clean_columns,axis=1)

# Ashton
k = 3
prefs = ['Bedroom','Bathroom','Car','Landsize', 'BuildingArea', 'Latitude','Longitude']

all_points = []
all_centroids = []

for pref in prefs:
    kmeansTrain = Kmeans.Kmeans(train_X, train_Y, k, pref)

    train_X, points, centroids = kmeansTrain.cluster()

    all_points.append(points)
    all_centroids.append(centroids)
    
    # Plot the data
    kmeansTrain.plotKmean(train_X, points, centroids, pref)

for pref in prefs:
    kmeansTest = Kmeans.Kmeans(test_X, test_Y, k, pref)

    test_X, points, centroids = kmeansTest.cluster()

    all_points.append(points)
    all_centroids.append(centroids)

    # Plot the data
    # kmeansTest.plotKmean(test_X, points, centroids, pref)

# kmeans = Kmeans.Kmeans(train_X, train_Y, k, prefs)

# train_X, points, centroids = kmeans.cluster()

# all_points.append(points)
# all_centroids.append(centroids)

# kmeans.plotKmean(train_X, points, centroids, prefs)

print('train_X columns')
print(train_X.columns)

print('test_X columns')
print(test_X.columns)






from performance_metrics import ElbowMethod
from DataProcessor import DataProcessor
import pandas as pd
from sklearn.impute import SimpleImputer

# Identify features to be used (excluding 'Type')
columnsToKeep = ['Price', 'Bedroom', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'Latitude', 'Longitude']

# Path to dataset
filepath = './dataset/Melbourne_housing_FULL.csv'

# Step 1: Load and preprocess the data
data_processor = DataProcessor()
data = data_processor.LoadData(filepath)

# Drop columns that are not relevant or cannot be converted to numeric
data = data_processor.keepSelectedColumns(columnsToKeep)

# Impute missing values with the mean of each column
imputer = SimpleImputer(strategy='mean')
data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Step 2: Use the ElbowMethod class with the preprocessed data
elbow_method = ElbowMethod(data)  # Initialize ElbowMethod
elbow_method.evaluate(max_clusters=10)
elbow_method.plot()
optimal_clusters = elbow_method.optimal_number_of_clusters()
# rint("Optimal number of clusters:", optimal_clusters)



# Data Preprocessing
from sklearn.preprocessing import StandardScaler
# from Kmeans import Kmeans
from performance_metrics import ElbowMethod

# Load and preprocess the data
data_processor = DataProcessor()
data = data_processor.LoadData(filepath)
data = data_processor.keepSelectedColumns(columnsToKeep)
imputer = SimpleImputer(strategy='mean')
data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Elbow Method for Optimal K
elbow_method = ElbowMethod(data_scaled)
elbow_method.evaluate(max_clusters=10)
elbow_method.plot()
optimal_clusters = elbow_method.optimal_number_of_clusters()
print("Optimal number of clusters:", optimal_clusters)
print(train_X.columns)
print(train_X)


# !!! temporary fix
pref = 'Bedroom'

from sklearn.metrics import silhouette_score

# Assuming kmeans.cluster() returns x_new, points, centroids
kmeans_instance = Kmeans.Kmeans(train_X, train_Y, optimal_clusters, pref)
x_new, points, centroids = kmeans_instance.cluster()
assignment = x_new['Assignment' + pref].values  # Extract the cluster assignments

silhouette_avg = silhouette_score(points, assignment)
print("Silhouette Score:", silhouette_avg)



from sklearn.metrics import silhouette_score

# Assuming kmeans.cluster() returns x_new, points, centroids
kmeans_instance = Kmeans(train_X, train_Y, optimal_clusters, pref)
x_new, points, centroids = kmeans_instance.cluster()
assignment = x_new['Assignment' + pref].values  # Extract the cluster assignments

silhouette_avg = silhouette_score(points, assignment)
print("Silhouette Score:", silhouette_avg)