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
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from PySide6.QtWidgets import QApplication

# ----------------------------------------------- DATA PROCESSING -----------------------------------------------

# Set training data size to 70%
train_size = 0.7

# Define features to be used
columnsToKeep = ['Price', 'Type', 'Bedroom', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'Latitude', 'Longitude']

# Path to the dataset
filepath = './dataset/Melbourne_housing_FULL.csv'

# Identify prediction column (ground truth)
prediction_column = 'Price'

# Create a DataProcessor object and load the data
dp_obj = DataProcessor.DataProcessor()
dp_obj.LoadData(filepath)

# Process data: keep selected columns, filter, shuffle, and reduce size
dp_obj.keepSelectedColumns(columnsToKeep)
dp_obj.filterMelbourneData()
dp_obj.remove_outliers(columnsToKeep, plot=False)
dp_obj.shuffleData()
allData = dp_obj.getData()  # for user interface at the end
# dp_obj.reduceDataSize(2100)  # Remove this line to train on the full dataset

# Split data into training and testing sets
train_X, train_Y, test_X, test_Y = dp_obj.splitData(train_size, prediction_column)

# Remove dwelling type and additional landsize/building area data for some processing
clean_columns = ['Type', 'Landsize']
train_X_clean = train_X.drop(columns=clean_columns, axis=1)
test_X_clean = test_X.drop(columns=clean_columns, axis=1)

# ----------------------------------------------- KMEANS CLUSTERING -----------------------------------------------

# Number of clusters
k = 3
prefs = ['Bedroom', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'Latitude', 'Longitude']

all_points = []
all_centroids = []

# Perform KMeans clustering for each preference in training data
for pref in prefs:
    kmeansTrain = Kmeans.Kmeans(train_X, train_Y, k, pref)
    train_X, points, centroids = kmeansTrain.cluster()
    all_points.append(points)
    all_centroids.append(centroids)

# Perform KMeans clustering for each preference in testing data
for pref in prefs:
    kmeansTest = Kmeans.Kmeans(test_X, test_Y, k, pref)
    test_X, points, centroids = kmeansTest.cluster()
    all_points.append(points)
    all_centroids.append(centroids)

kmeansTrain.pca(prefs)

kmeansTrain.tSNE(prefs)

# ----------------------------------------------- NON-LINEAR REGRESSION -----------------------------------------------

# Define features for training (ordered by correlation to price)
featuresToTrain = ['Latitude', 'BuildingArea', 'Longitude', 'Bathroom', 'Bedroom']

nlr = NLRegression.NLRegression(train_X, train_Y, test_X, test_Y, featuresToTrain)
filename_ = 'nlr_model_R_2_0.694415207533843.pkl'
filepath_ = f'./saved_models/{filename_}'
nlr.loadModel(filepath_)

# ----------------------------------------------- USER INTERFACE -----------------------------------------------

# Calculate mean values if no user preference is provided
meanLand = dp_obj.getAverage('Landsize')
meanBuilding = dp_obj.getAverage('BuildingArea')

# Instantiate UserInterface and run the GUI
app = QApplication(sys.argv)
user_interface = UserInterface.UserInterface()
user_interface.gui_inputs(allData)
user_interface.show()
app.exec()

# Retrieve and unpack the result after the event loop ends
result = user_interface.get_result()

if result:
    bedrooms, bathrooms, car, landsize, buildingarea, latitude, longitude = result
    print(" ")
    print("Bedrooms =", bedrooms, "Bathrooms =", bathrooms, "Car parks =", car, "Land size =", landsize, "Building area =", buildingarea, "Latitude =", latitude, "Longitude =", longitude)

    # Predict actual value using Non-Linear Regression
    sample = pd.Series([latitude, buildingarea, longitude, bathrooms, bedrooms], index=featuresToTrain)
    pred, bounds = nlr.predictActual(sample)

    pred_rounded = pred.round(2)[0]

    lowerBound = bounds[0].round(2)[0]
    upperBound = bounds[1].round(2)[0]

    print(" ")
    print("Prediction: $", pred_rounded)
    print("Min:     $", lowerBound)
    print("Max:     $", upperBound)

    houses = user_interface.get_closest_houses(allData, pred_rounded, latitude, longitude, buildingarea, bathrooms, bedrooms, car, n=5, show_window=True)

    # Print the houses
    # print(" ")
    # print("Existing houses with closest features: ")
    # print(houses)
else:
    print("No input received from the user.")
