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
### @Lucas
Detailed comments are provided in the `NLRegression` class, located in the `NLRegression.py` file.

## Evaluation Metrics
### @Wajeeha
Detailed comments are provided in the `ElbowMethod` class, located in the `performance_metrics.py` file.

## User Interface
For operation of the algorithm, a user interface was developed that accepts user preferences for the value of each feature. Detailed comments are provided in the `UserInterface` class, located in the `UserInterface.py` file.

## Prediction
### Linear Regression
A prediction is made based on the weights trained for each cluster (house category). The prediction returns a range:
- The lower value represents the expected price for an economic house with the specified features.
- The higher value represents the expected price for a luxury house with the specified features.

### Gaussian Processing
### @Lucas
Comments and details are provided in the relevant sections of the documentation.

## Integration
For operation of the algorithm, an integrated script `estimator` is used. Detailed comments are provided in the `estimator.py` and `estimator.ipynb` files.
