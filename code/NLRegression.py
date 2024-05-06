# Contributions - Lucas

# implmenting gaussian process regression

import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class NLRegression:
    def __init__(self, train_X, train_Y, test_X, test_Y, featuresToTrain ,scale_data=True, kernel=None):
        """
        Initialize the NLRegression model with training and testing data.
        Parameters:
        - train_X, train_Y: Training features and targets.
        - test_X, test_Y: Testing features and targets.
        - scale_data: Boolean, whether to scale data using StandardScaler.
        - kernel: scikit-learn kernel object. If None, default kernel is used.
        """

        train_X = train_X[featuresToTrain]
        test_X = test_X[featuresToTrain]

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

        # Normalize the data if needed
        self.scaler = StandardScaler() if scale_data else None
        if scale_data:
            train_X = pd.DataFrame(self.scaler.fit_transform(train_X), columns=train_X.columns)
            test_X = pd.DataFrame(self.scaler.transform(test_X), columns=test_X.columns)

        self.train_X = train_X
        self.train_Y = train_Y
        self.test_X = test_X
        self.test_Y = test_Y

        # look at using different kernels

        if kernel is None or not isinstance(kernel, (RBF, C)):
            """
            The constant kernel represents a constant function that predicts the mean of the target variable.
            The RBF kernel represents a stationary kernel that models the covariance of the target variable.
            """
            constant_kernel = C(1.0, (1e-2, 1e+3)) # constant kernel, default value 1.0, bounds (1e-4, 1e+2)
            rbf_kernel = RBF(1.0, (1e-3, 1e+3)) # RBF kernel, default value 1.0, bounds (1e-5, 1e+2)

            self.kernel = constant_kernel * rbf_kernel
        else:
            self.kernel = kernel

    def train(self):
        """Train the Gaussian Process Regressor."""
        self.gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=10)
        self.gp.fit(self.train_X, self.train_Y)

    def predict(self, input_data=None):
        """Predict using the trained Gaussian Process Regressor."""
        y_pred, sigma = self.gp.predict(input_data, return_std=True)
        return y_pred, sigma

    # !!! this will be removed once performance metrics class is implemented
    def evaluate(self):
        """Evaluate the model using the test data."""
        y_pred, sigma = self.predict(self.test_X)
        score = self.gp.score(self.test_X, self.test_Y)
        print("R² Score:", score)
        return score
    
    def plot(self):
        # Get predictions for the test data
        y_pred, sigma = self.predict(self.test_X)

        # Assuming the first column of test_X is the intended x-axis values
        x = self.test_X.iloc[:, 0].reset_index(drop=True)  # Reset index and drop the old one

        # Reset index for self.test_Y before creating DataFrame
        y = self.test_Y.reset_index(drop=True)

        # Create a DataFrame from x, y, y_pred, and sigma
        data = pd.DataFrame({
            'x': x,
            'y': y,
            'y_pred': y_pred,
            'sigma': sigma
        })

        print(data)

        # Sort the DataFrame based on x values
        data_sorted = data.sort_values(by='x')

        # Plotting
        plt.figure()
        plt.plot(data_sorted['x'], data_sorted['y'], 'r:', label='Actual')
        plt.plot(data_sorted['x'], data_sorted['y_pred'], 'b-', label='Predicted')

        # can't see this, could be the scale of the data
        plt.fill_between(data_sorted['x'],
                 data_sorted['y_pred'] - 1.96,
                 data_sorted['y_pred'] + 1.96,
                 alpha=0.5, facecolor='b', edgecolor='None', label='95% confidence interval')
        plt.xlabel('Feature')
        plt.ylabel(self.test_Y.name) # base label name of y test data name
        plt.title('Gaussian Process Regression')
        plt.legend()
        plt.show()


