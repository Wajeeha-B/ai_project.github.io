# Contributions - Lucas

# implmenting gaussian process regression

import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from numpy import sqrt, diag

class NLRegression:
    def __init__(self, train_X, train_Y, test_X, test_Y, featuresToTrain):
        """
        Initialize the NLRegression model with training and testing data.
        Parameters:
        - train_X, train_Y: Training features and targets.
        - test_X, test_Y: Testing features and targets.
        - featuresToTrain: List of features to be used.
        - kernel_type: Type of kernel to use for the Gaussian Process Regressor.
        """

        train_X = train_X[featuresToTrain]
        test_X = test_X[featuresToTrain]

        # train_X = train_X[featuresToTrain]
        # test_X = test_X[featuresToTrain]

        # Check if the input data is valid
        if any(v is None for v in [train_X, train_Y, test_X, test_Y]):
            raise ValueError("Please provide training and test data")
        if len(train_X) != len(train_Y):
            raise ValueError("Training data and target data should have the same length")
        if len(test_X) != len(test_Y):
            raise ValueError("Test data and target data should have the same length")
        
        if not isinstance(train_X, (pd.DataFrame)):
            raise TypeError("train_X must be a pandas DataFrame")
        if not isinstance(train_Y, (pd.Series)):
            raise TypeError("train_Y must be a pandas Series")
        if train_X.ndim != 2 or train_Y.ndim != 1:
            raise ValueError("train_X must be two-dimensional and train_Y must be one-dimensional")
        
        if not isinstance(test_X, (pd.DataFrame)):
            raise TypeError("test_X must be a pandas DataFrame")
        if not isinstance(test_Y, (pd.Series)):
            raise TypeError("test_Y must be a pandas Series")
        if test_X.ndim != 2 or test_Y.ndim != 1:
            raise ValueError("test_X must be two-dimensional and test_Y must be one-dimensional")
        
        # One-hot encoding of discrete features
        train_X = pd.get_dummies(train_X)
        test_X = pd.get_dummies(test_X)

        # Align test columns with train columns
        train_X, test_X = train_X.align(test_X, join='left', axis=1, fill_value=0)

        # Initialize scalers and apply scaling to features and targets
        # Need to remember to inverse transform the target when predicting

        self.scaler_features = StandardScaler()
        
        self.train_X = pd.DataFrame(self.scaler_features.fit_transform(train_X), columns=train_X.columns)

        self.test_X = pd.DataFrame(self.scaler_features.transform(test_X), columns=test_X.columns)

        self.scaler_target = StandardScaler()
        self.train_Y = pd.Series(self.scaler_target.fit_transform(train_Y.values.reshape(-1, 1)).flatten(), name=train_Y.name)
        self.test_Y = pd.Series(self.scaler_target.transform(test_Y.values.reshape(-1, 1)).flatten(), name=test_Y.name)

        # plot each scaled feature against the target
        for feature in self.train_X.columns:
            plt.scatter(self.train_X[feature], self.train_Y)
            plt.xlabel(feature)
            plt.ylabel(self.train_Y.name)
            plt.show()


        """
        The constant kernel represents a constant function that predicts the mean of the target variable.
        The RBF kernel represents a stationary kernel that models the covariance of the target variable.
        """
        constant_kernel = ConstantKernel(1.0, (1e-2, 1e+3))
        rbf_kernel = RBF(1.0, (1e-3, 1e+3))
        self.kernel = constant_kernel * rbf_kernel

    def scaleData(self, data):
        """Scale the input data using the trained scalers."""
        if isinstance(data, pd.Series):
            # Reshape the data and retain the original index as columns
            data_df = pd.DataFrame([data.values], columns=data.index)
            scaled = pd.DataFrame(self.scaler_features.transform(data_df), columns=data.index)
        else:
            # Directly transform if it's already a DataFrame
            scaled = pd.DataFrame(self.scaler_features.transform(data), columns=data.columns)

        return scaled

    def train(self):
        """Train the Gaussian Process Regressor."""
        self.gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=10) # optimizer='fmin_l_bfgs_b')
        
        self.gp.fit(self.train_X, self.train_Y)
        self.evaluate()

    def predict(self, input_data=None):
        """Predict using the trained Gaussian Process Regressor."""
        y_pred, std = self.gp.predict(input_data, return_std=True)
        return y_pred, std
    
    def predictActual(self, input_data=None):
        """Predict using the trained Gaussian Process Regressor and return predictions with uncertainty."""
        # Ensure input data is scaled correctly
        scaled_data = self.scaleData(input_data)

        # Make predictions using the Gaussian Process model
        y_pred, std = self.gp.predict(scaled_data, return_std=True)

        # Inverse transform the predictions back to the original scale
        y_pred = self.scaler_target.inverse_transform(y_pred.reshape(-1, 1)).flatten()

        # Inverse transform the standard deviation as an estimate of uncertainty
        # Note: StandardScaler does not directly affect standard deviations, thus apply sqrt and diag
        std = sqrt(diag(std))

        return y_pred, std

    def evaluate(self):
        """Evaluate the model using the test data."""
        score = self.gp.score(self.test_X, self.test_Y)
        print("R² Score:", score)
        return score
    
    def plot(self):
        """Plot the actual vs. predicted values with standard deviation."""
        # Get predictions and confidence intervals
        y_pred, std = self.predict(self.test_X)

        # Ensure all data is converted to numpy arrays
        x = self.test_X.iloc[:, 0].values.flatten()
        y = self.test_Y.values.flatten()

        data = pd.DataFrame({
            'x': x,
            'y': y,
            'y_pred': y_pred,
            'lower_bound': y_pred - std,
            'upper_bound': y_pred + std
        })

        # Sort data based on x values and ensure numpy arrays
        data_sorted = data.sort_values(by='x')
        x_vals = data_sorted['x'].values
        y_vals = data_sorted['y'].values
        y_pred_vals = data_sorted['y_pred'].values
        lower_vals = data_sorted['lower_bound'].values
        upper_vals = data_sorted['upper_bound'].values

        # Plot using an alternative function (scatter or step)
        plt.figure(figsize=(10, 6))
        plt.scatter(x_vals, y_vals, color='red', label='Actual')
        plt.plot(x_vals, y_pred_vals, color='blue', label='Predicted')
        plt.fill_between(x_vals,
                        lower_vals,
                        upper_vals,
                        alpha=0.2, facecolor='blue', label='std')
        plt.xlabel('Feature')
        plt.ylabel(self.test_Y.name or 'Target')
        plt.title('Gaussian Process Regression with Standard Deviation')
        plt.legend()
        plt.grid(True)
        plt.show()


    def cross_validate(self, num_folds=3, n_jobs=-1):
        """Perform cross-validation with parallel processing."""
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        scores = cross_val_score(self.gp, self.train_X, self.train_Y, cv=kfold, scoring='r2', n_jobs=n_jobs)
        print(f"Cross-Validation Scores (R²): {scores}")
        print(f"Mean R²: {scores.mean()}, Standard Deviation: {scores.std()}")

    