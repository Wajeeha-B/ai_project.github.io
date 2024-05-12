# Contributions - John
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LinearRegression
class LRegression:
    def __init__(self):
        pass

    def LinReg(self,x,y,values,assignments,k):
        # Initialise weights
        self.w = []

        # Calculate weights using linear regression
        # @NOTE: Based on 'value' (feature) and corresponding assignment
        for value,assignment in zip(values,assignments):
            # Check if targeted heading is within the columns, else skip
            if value not in x.columns or assignment not in x.columns:
                continue
            x1 = x.loc[:,value][x[assignment] == k]
            y1 = y[x[assignment] == k]
            x1 = x1.values.reshape(-1,1)
            y1 = y1.values.reshape(-1,1)
            x1t = np.transpose(x1)
            x1tx = np.dot(x1t,x1)
            x1tx_inv = np.linalg.inv(x1tx)
            x1ty = np.dot(x1t,y1)
            weight = np.dot(x1tx_inv,x1ty)
            self.w.append(weight)
        return self.w
    
    def MSE(self,x,w,y):
        # Calculate prediction (sum of weights)
        y_pred = x @ w

        # Take average prediction of the weights
        y_pred = y_pred / w.shape[1]

        # Calculate MSE
        self.y_mse = metrics.mean_squared_error(y,y_pred)

        # Return MSE
        return self.y_mse
    
    def predict(self,features,w):
        # Calculate prediction (sum of weights)
        self.pred = features @ w

        # Take average prediction of the weights
        self.pred = self.pred / w.shape[1]

        # Return prediction
        return self.pred