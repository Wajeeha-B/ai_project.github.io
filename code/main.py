# Contributions - everyone 

import sys
import DataProcessor
sys.path.append('code')


# ----------------parameters------------------
train_size = 0.8 # 80% training data
# columnsToKeep = ['Type', 'Price', 'Distance', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea']

# !!! reducing featues for testing
columnsToKeep = ['Price', 'Size']


filepath = 'dataset/Melbourne_housing_FULL.csv'
prediction_column = 'Price'
# --------------------------------------------

# ---------------Process Data-----------------
dp_obj = DataProcessor.DataProcessor()
dp_obj.LoadData(filepath)

dp_obj.filterData()
dp_obj.keepSelectedColumns(columnsToKeep)
dp_obj.shuffleData()

# !!! reduce data size for testing
dp_obj.reduceDataSize(100)

train_X, train_Y, test_X, test_Y = dp_obj.splitData(train_size, prediction_column)
# --------------------------------------------
# ------------kmeans Clustering---------------
from kmeans import kmeans
centroids, assignment = kmeans(train_X, train_Y, columnsToKeep[1])
print('clustering finished')
# --------------------------------------------
# ---------------Train Model------------------
import NLRegression
nlr = NLRegression.NLRegression(train_X, train_Y, test_X, test_Y, scale_data=False)
nlr.train()
# --------------------------------------------
# -------------Evaluate Model-----------------
nlr.evaluate()

# --------------------------------------------
# -------------Visualize Model----------------
nlr.plot()
# --------------------------------------------

