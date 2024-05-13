# ai_project.github.io

<!-- Overview -->
This project uses Machine Learning (ML) techniques to estimate house prices in Melbourne based on a range of user-input features.

<!-- Features -->
The features include:
> Number of bedrooms
> Number of bathrooms
> Number of parking spaces
> Landsize (sqm)
> Building area (sqm)
> Latitude (general area, north or south)
> Longitude (general area, east or west)

<!-- Dataset -->
The dataset uses nearly 35,000 dwelling sales from across Melbourne, VIC, Australia. It was sourced from the below:
@TODO: insert link

<!-- Processing -->
The data is read in and processed using the DataProcessor class. This class narrows down the dataset to only the target features, before removing data points missing key values.

<!-- K-Means -->
@Ashton

<!-- Linear Regression -->
Based on the clusters of each feature, linear regression is completed to calculate the weights.
Details are provided in the LRegression class, located in the 'LRegression.py' file.

<!-- Gaussian Processing (Non-Linear Regression) -->
@Lucas

<!-- Evaluation Metrics -->
@Wajeeha

<!-- User Interface -->
For operation of the algorithm, a user interface was developed that accepts user preferences for the value of each feature.
If an input is unknown, a '0' may be input and a default value will be provided.

<!-- Prediction -->
@John & @Lucas

<!-- Integration -->
@Anyone