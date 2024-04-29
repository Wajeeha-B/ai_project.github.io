# import other files in to this file and run the code

import sys
sys.path.append('code')
import DataProcessor

filepath = 'dataset/Melbourne_housing_FULL.csv'
dp_obj = DataProcessor.DataProcessor()
data = dp_obj.LoadData(filepath)

columnsToKeep = ['Type', 'Price', 'Distance', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea']
data = dp_obj.keepSelectedColumns(columnsToKeep)

data = dp_obj.randomiseData()

data = dp_obj.dataFilter()

print("size of dataset: ", data.shape)

# print(data.iloc[10])


data_split = 0.8 # 80% training data
training_data,test_data = dp_obj.splitData(data_split)


# print(data.iloc[10])

print("training size", len(training_data))
print("test size", len(test_data))
