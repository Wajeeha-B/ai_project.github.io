# get data from dataset folder

# Contributions - everyone 

import os
import pandas as pd

class DataProcessor:
    def __init__(self):
        print("Data Processor object created")
        pass
    
    def LoadData(self, filepath):
        self.data = pd.DataFrame()
        self.filepath = filepath
        if os.path.exists(self.filepath):
            self.data = pd.read_csv(self.filepath)
        else:
            print("File not found")

        return self.data
    
    # provide a list of columns to drop
    def dropUnwantedColumns(self, columns):
        self.data = self.data.drop(columns, axis=1)
        return self.data
    
    def keepSelectedColumns(self, columns):
        self.data = self.data[columns]
        return self.data
    
    def randomiseData(self):
        # From Quiz 1
        self.data = self.data.sample(frac=1)
        return self.data
    
    def dataFilter(self):
        # dropping empty columns
        self.data.dropna(subset=['Price'], inplace=True)
        self.data.dropna(subset=['Distance'], inplace=True)
        self.data.dropna(subset=['Bathroom'], inplace=True)

        self.data['Type'] = self.data['Type'].replace('t', 'h')

        self.data = self.data.fillna({'Car': 0})

        self.data = self.data[self.data['Bathroom'] >= 1]
        self.data = self.data[self.data['Bedroom2'] < 10]

        self.data['Size'] = self.data['Landsize']

        self.data.loc[self.data['Type'] == 'u', 'Size'] = self.data['BuildingArea']

        self.data = self.data.drop(columns=['Landsize', 'BuildingArea'])

        return self.data

        
    
    def splitData(self, ratio):
        # From Quiz 1, split into training & test
        self.train_data = self.data.sample(frac=ratio)
        self.test_data = self.data.drop(self.train_data.index)
        return self.train_data, self.test_data