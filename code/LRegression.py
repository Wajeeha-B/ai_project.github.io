# Contributions - John
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LinearRegression
class LRegression:
    def __init__(self):
        pass

    def LinReg(self,features,target,assignment):
        # Features to be used for regression
        self.features = features

        # Target for regression
        self.target = target

        # Cluster assignment
        self.assignment = assignment

        self.y_preds = {}

        # for i in range(self.features.shape[0]):
        #     if self.features['Assignment'][i] != assignment[i]:
        #         self.features.drop(i,axis=0)
        #         self.target.drop(i,axis=0)

        w = []

        for i in range(max(self.features['Assignment'])):
            x1 = self.features[self.assignment == i]
            y = self.target[self.assignment == i]
            w.append(np.linalg.inv(np.transpose(x1) @ x1) @ x1.T @ y)
            y_pred = x1 @ w[i]
            self.y_preds[i] = y_pred

        print(self.features.shape)

        # Assign inputs for calculation
        # x1 = self.features
        # y = self.target

        # Calculate weights
        # self.w = np.linalg.inv(np.transpose(x1) @ x1) @ x1.T @ y

        # Make prediction
        # y_pred = x1 @ self.w
        # self.y_pred = y_pred.reshape(-1,1)

        # Return prediction
        return self.y_preds
    
    def LinearRegression(self,features,target):
        for i in range(features['Assignment'].shape[0]):
            x1 = features[features['Assignment'] == i]
            x1 = x1.drop('Assignment',axis=1)
            y = target[target['Assignment'] == i]

            # w = np.linalg.inv(np.transpose(x1) @ x1) @ x1.T @ y
            x1t = np.transpose(features)
            x1tx = np.dot(x1t,x1)
            x1tx_inv = np.linalg.inv(x1tx)
            x1ty = np.dot(x1t,y)
            w = np.dot(x1tx_inv,x1ty)

            self.y_pred = x1 @ w
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
    
    def predict(self,features):
        pred = features @ self.w
        return pred.reshape(-1,1)