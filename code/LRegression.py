# Contributions - John
import numpy as np
from sklearn import metrics
class LRegression:
    def __init__(self):
        pass

    def LinReg(self,features,target):
        # Features to be used for regression
        self.features = features

        # Target for regression
        self.target = target

        # Assign inputs for calculation
        x1 = self.features
        y = self.target

        # Calculate weights
        self.w = np.linalg.inv(np.transpose(x1) @ x1) @ x1.T @ y

        # Make prediction
        y_pred = x1 @ self.w
        self.y_pred = y_pred.reshape(-1,1)

        # Return prediction
        return self.y_pred
    
    def MSE(self):
        # Target value
        self.target

        # Predicted value
        self.y_pred

        # Calculate MSE
        self.y_mse = metrics.mean_squared_error(self.target,self.y_pred)

        # Return MSE
        return self.y_mse
