# AI Project Documentation

## Collaborators
- **Ashton Powell** | 13915196
- **John Hunter** | 13938838
- **Lucas Moore** | 14173177
- **Wajeeha Batool** | 14279261

## Prepared For
- **Dr Raphael Falque**
- **Dr Fred Sukkar**
- **Monisha Uttsha**
- **Nadim Haque**
- **41118 Artificial Intelligence in Robotics**
- **Autumn 2024**
- **University of Technology Sydney (UTS)**

## Overview
This project utilises Machine Learning (ML) techniques to estimate house prices in Melbourne based on a range of user-input features.

## Features
The features include:
- Number of bedrooms
- Number of bathrooms
- Number of parking spaces
- Landsize (sqm)
- Building area (sqm)
- Latitude (general area, north or south)
- Longitude (general area, east or west)

## Dataset
The dataset comprises nearly 35,000 dwelling sales from across Melbourne, VIC, Australia. It was sourced from the following:
[Melbourne Housing Dataset](https://github.com/erkansirin78/datasets/blob/master/Melbourne_housing_FULL.csv)

## Processing
The data is read in and processed using the `DataProcessor` class. This class processes the dataset to include only the target features, removing data points missing key values.

## K-Means
### @Ashton
Detailed comments are provided in the `KMeans` class, located in the `Kmeans.py` file.

## Linear Regression
Based on the clusters of each feature, linear regression is completed to calculate the weights. Detailed comments are provided in the `LRegression` class, located in the `LRegression.py` file.

## Gaussian Processing (Non-Linear Regression)
The `NLRegression` class implements Gaussian Process Regression for predicting house prices. This class provides functionality for data scaling, model training, prediction, evaluation, and saving/loading the model. It also includes capabilities for feature importance analysis and interaction term identification using a Random Forest Regressor.

### Key Features

- **Initialisation**: The class is initialised with training and testing datasets, ensuring data validity and scaling.
- **Data Scaling**: Utilises `StandardScaler` for feature and target scaling, ensuring consistent data ranges for model training and prediction.
- **Gaussian Process Regression**: Implements Gaussian Process Regression with a combination of constant and rational quadratic kernels for robust predictions.
- **Prediction**: Provides methods for making predictions with associated uncertainty bounds, both scaled and unscaled.
- **Evaluation**: Evaluates model performance using metrics such as RMSE, MAE, and R² score.
- **Cross-Validation**: Includes functionality for cross-validation with parallel processing.
- **Model Persistence**: Supports saving and loading the trained model using pickle.
- **Feature Importance**: Trains a Random Forest Regressor to identify important features and potential interactions.
- **Interaction Terms**: Identifies and creates interaction terms between the most important features.


## Evaluation Metrics
### @Wajeeha
Detailed comments are provided in the `ElbowMethod` class, located in the `performance_metrics.py` file.

## User Interface
The `UserInterface` class is designed to provide a graphical interface for users to input their preferences for house features. This interface supports both console-based inputs and a graphical slider-based input system using the PySide6 library.

### Console-Based Input Method
- **Bedrooms**: Prompts the user to input the preferred number of bedrooms.
- **Bathrooms**: Prompts the user to input the preferred number of bathrooms.
- **Car Spaces**: Prompts the user to input the preferred number of car parking spaces.
- **Land Size**: Prompts the user to input the preferred land size (sqm), with an option to use the average size if unknown.
- **Building Area**: Prompts the user to input the preferred building area (sqm), with an option to use the average size if unknown.
- **Latitude**: Prompts the user to specify a preference for 'north' or 'south'.
- **Longitude**: Prompts the user to specify a preference for 'east' or 'west'.

### Graphical User Interface (GUI) Method
- **Window Setup**: The GUI window titled "User Preferences" allows users to set their preferences through sliders.
- **Sliders for Features**: Sliders are provided for each feature, with default values set to the average of the dataset and ranges based on the dataset's minimum and maximum values.
    - Bedrooms
    - Bathrooms
    - Car Spaces
    - Land Size
    - Building Area
    - Latitude
    - Longitude
- **Submit Button**: A submit button collects the user's selections and closes the interface.

### Closest Houses Feature
- **Similarity Calculation**: Calculates the difference between the user's preferred features and actual house features in the dataset, providing the top N closest matches.
- **Display Results**: Optionally displays a window listing the closest matching houses with their details.

This interface provides a user-friendly way to input preferences and receive predictions, enhancing the accessibility and usability of the house price estimation model.


## Prediction
### Linear Regression
A prediction is made based on the weights trained for each cluster (house category). The prediction returns a range:
- The lower value represents the expected price for an economic house with the specified features.
- The higher value represents the expected price for a luxury house with the specified features.

### Gaussian Processing
Gaussian Process Regression (GPR) predicts house prices with associated uncertainty bounds. The output includes:
- **Mean Prediction**: The average predicted price for the specified features.
- **Uncertainty Bounds**: The prediction interval, provided as a range around the mean prediction, indicating confidence in the predicted price.

The GPR model provides more nuanced predictions by incorporating the variability in the data, offering both a central price estimate and a measure of the uncertainty in this estimate. Comments and details are provided in the relevant sections of the documentation.


## Integration
For operation of the algorithm, an integrated script `estimator` is used. Detailed comments are provided in the `estimator.py` and `estimator.ipynb` files.

## Final usage:
To use a graphical user interface with a pre-trained model, the `GUI_estimator.py` executable is provided. A window with sliders for each feature will pop up when this script is run. These sliders are initialised to the average value for each feature and limited to the minimum and maximum values of the trained dataset, allowing users to adjust values as needed. Users who are unsure about a specific feature can leave the slider at its default average value.
