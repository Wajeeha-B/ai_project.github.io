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
# dp_obj.reduceDataSize(100)  # Remove this line to train on the full dataset

# Calculate correlation for selected columns
columnsToCorrelate = ['Price', 'Bedroom', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'Latitude', 'Longitude']
dp_obj.calculateCorelation(columnsToCorrelate)

# Plot data distribution
dp_obj.plotDataNormalDistribution()

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
    kmeansTrain.plotKmean(train_X, points, centroids, pref)

# Perform KMeans clustering for each preference in testing data
for pref in prefs:
    kmeansTest = Kmeans.Kmeans(test_X, test_Y, k, pref)
    test_X, points, centroids = kmeansTest.cluster()
    all_points.append(points)
    all_centroids.append(centroids)

# Generate and plot PCA
kmeansTrain.pca(prefs)
kmeansTrain.plotPCA()

# Generate and plot t-SNE
kmeansTrain.tSNE(prefs)
kmeansTrain.plotTSNE()

# ----------------------------------------------- LINEAR REGRESSION -----------------------------------------------

# Train Linear Regression model
lr = LRegression.LRegression()
values = ['Bedroom', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'Latitude', 'Longitude']
assignments = ['AssignmentBedroom', 'AssignmentBathroom', 'AssignmentCar', 'AssignmentLandsize', 'AssignmentBuildingArea', 'AssignmentLatitude', 'AssignmentLongitude']

# Initialise weights and MSE for training
w = []
train_mse = []

# Calculate weights for each cluster
for i in range(k):
    weight = lr.LinReg(train_X, train_Y, values, assignments, i)
    w.append(weight)
w = np.array(w).reshape(len(w), len(w[0]), -1)

# Calculate MSE for training data
for i in range(k):
    train_mse.append(lr.MSE(train_X[values], w[i, :], train_Y))
print("Training error (houses) = ", train_mse)

# Test Linear Regression model
test_mse = []

# Calculate MSE for testing data
for i in range(k):
    test_mse.append(lr.MSE(test_X[values], w[i, :], test_Y))
print("Testing error (houses) = ", test_mse)

# ----------------------------------------------- NON-LINEAR REGRESSION -----------------------------------------------

# Define features for training (ordered by correlation to price)
featuresToTrain = ['Latitude', 'BuildingArea', 'Longitude', 'Bathroom', 'Bedroom']

# Check if a new model should be trained
query = input("Do you want to train a new model? (y/n): ")
if query == 'y':
    nlr = NLRegression.NLRegression(train_X, train_Y, test_X, test_Y, featuresToTrain)
    nlr.train()
    # Save the model if requested
    query = input("Do you want to save the model? (y/n): ")
    if query == 'y':
        _, _, r_2 = nlr.evaluate()
        filepath_ = f'./saved_models/nlr_model_R_2_{r_2}'
        nlr.saveModel(filepath_)
else:
    nlr = NLRegression.NLRegression(train_X, train_Y, test_X, test_Y, featuresToTrain)
    filename_ = input("Enter the filename of the model to load: ")
    filepath_ = f'./saved_models/{filename_}'
    nlr.loadModel(filepath_)

# Plot predictions vs expected values
nlr.plot()

# Perform cross-validation
print("Would you like to cross-validate? ")
query = input("Enter 'y' for yes or 'n' for no: ")
if query == 'y':
    nlr.cross_validate()

# -------------------------------------------- EVALUATION METRICS --------------------------------------------

# Define features to be used (excluding 'Type')
columnsToKeep = ['Price', 'Bedroom', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'Latitude', 'Longitude']

# Path to dataset
filepath = './dataset/Melbourne_housing_FULL.csv'

# Load and preprocess the data
data_processor = DataProcessor.DataProcessor()
data = data_processor.LoadData(filepath)
data = data_processor.keepSelectedColumns(columnsToKeep)
imputer = SimpleImputer(strategy='mean')
data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Elbow Method for Optimal K
elbow_method = performance_metrics.ElbowMethod(data_scaled)
elbow_method.evaluate(max_clusters=10)
elbow_method.plot()
optimal_clusters = elbow_method.optimal_number_of_clusters()
print("Optimal number of clusters:", optimal_clusters)

# Perform KMeans clustering with optimal number of clusters
kmeans_instance = Kmeans.Kmeans(train_X, train_Y, optimal_clusters, pref)
pref = 'Car'  # Temporary fix
x_new, points, centroids = kmeans_instance.cluster()
assignment = x_new['Assignment' + pref].values  # Extract the cluster assignments

# Calculate silhouette score
silhouette_avg = performance_metrics.calculate_silhouette_score(points, assignment)
print("Silhouette Score:", silhouette_avg)

# ----------------------------------------------- USER INTERFACE -----------------------------------------------

# Calculate mean values if no user preference is provided
meanLand = dp_obj.getAverage('Landsize')
meanBuilding = dp_obj.getAverage('BuildingArea')

# Instantiate UserInterface and run the GUI
app = QApplication(sys.argv)
user_interface = UserInterface.UserInterface()
user_interface.gui_inputs(allData)
user_interface.show()
app.exec_()

# Retrieve and unpack the result after the event loop ends
result = user_interface.get_result()

if result:
    bedrooms, bathrooms, car, landsize, buildingarea, latitude, longitude = result
    print("Bedrooms =", bedrooms, "Bathrooms =", bathrooms, "Car parks =", car, "Land size =", landsize, "Building area =", buildingarea, "Latitude =", latitude, "Longitude =", longitude)

    # Return predicted value from Linear Regression

    # Assign feature inputs
    targets = [bedrooms, bathrooms, car, landsize, buildingarea, latitude, longitude]

    # Make prediction for each level
    pred = lr.predict(targets, w)

    # Return minimum and maximum predicted values
    print("Minimum value = $", np.min(pred).round(2))
    print("Maximum value = $", np.max(pred).round(2))

    # Predict actual value using Non-Linear Regression
    sample = pd.Series([latitude, buildingarea, longitude, bathrooms, bedrooms], index=featuresToTrain)
    pred, bounds = nlr.predictActual(sample)

    pred_rounded = pred.round(2)[0]

    lowerBound = bounds[0].round(2)[0]
    upperBound = bounds[1].round(2)[0]

    print("Prediction: $", pred_rounded)
    print("Min:     $", lowerBound)
    print("Max:     $", upperBound)

    houses = user_interface.get_houses_within_range(allData, pred_rounded, latitude, longitude, buildingarea, bathrooms, bedrooms, car)

    # Print the houses
    print("Existing houses with given features: ")
    print(houses)
else:
    print("No input received from the user.")
