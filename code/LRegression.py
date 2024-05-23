# Contributions - John

# Import libraries
import numpy as np # For linear algebra
from sklearn import metrics # For Mean Squared Error

# Define class
class LRegression:
    # Initialise with a pass to return immediately
    def __init__(self):
        pass

    # Define function for calculating weights, accepting necessary arguments
    def LinReg(self,x,y,values,assignments,k):
        # Initialise weights as an array
        self.w = []

        # Calculate weights
        # @NOTE: Based on 'value' (feature) and corresponding assignment
        for value,assignment in zip(values,assignments):
            # Check if targeted heading is within the DataFrame columns, else skip
            if value not in x.columns or assignment not in x.columns:
                continue
            
            # Calculate weight for the feature
            x1 = x.loc[:,value][x[assignment] == k]
            y1 = y[x[assignment] == k]
            x1 = x1.values.reshape(-1,1)
            y1 = y1.values.reshape(-1,1)
            x1t = np.transpose(x1)
            x1tx = np.dot(x1t,x1)
            x1tx_inv = np.linalg.inv(x1tx)
            x1ty = np.dot(x1t,y1)
            weight = np.dot(x1tx_inv,x1ty)

            # Append weight to array
            self.w.append(weight)
        
        # Return weights for cluster 'k'
        return self.w
    
    # Define function for calculating Mean Squared Error, accepting necessary arguments
    def MSE(self,x,w,y):
        # Calculate prediction (sum of weights)
        y_pred = x @ w

        # Refine prediction based on weights
        y_pred = y_pred / w.shape[1]

        # Calculate MSE using sklearn
        self.y_mse = metrics.mean_squared_error(y,y_pred)

        # Return MSE
        return self.y_mse
    
    # Define function for making Linear Regression prediction, accepting necessary arguments
    def predict(self,features,w):
        # Calculate prediction (sum of weights)
        self.pred = features @ w

        # Refine prediction based on weights
        self.pred = self.pred / w.shape[1]

        # Return prediction
        return self.pred