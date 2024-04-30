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
    
    def shuffleData(self):
        # From Quiz 1
        self.data = self.data.sample(frac=1).reset_index(drop=True)
        return self.data
    
    def filterData(self):
        # dropping empty columns

        # check it exists before dropping to avoid errors
        if 'Price' in self.data.columns:
            self.data.dropna(subset=['Price'], inplace=True)
            self.data = self.data[self.data['Price'] > 0]

        
        if 'Distance' in self.data.columns:
            self.data.dropna(subset=['Distance'], inplace=True)

        if 'Bathroom' in self.data.columns:
            self.data.dropna(subset=['Bathroom'], inplace=True)
            self.data = self.data[self.data['Bathroom'] >= 1]
        
        if 'Type' in self.data.columns:
            self.data['Type'] = self.data['Type'].replace('t', 'h')
            
            if 'Landsize' in self.data.columns and 'BuildingArea' in self.data.columns:
                self.data = self.data.dropna(subset=['Landsize', 'BuildingArea'])

                self.data['Size'] = self.data['Landsize']
                self.data.loc[self.data['Type'] == 'u', 'Size'] = self.data['BuildingArea']
                self.data = self.data.drop(columns=['Landsize', 'BuildingArea'])

        if 'Car' in self.data.columns:
            self.data = self.data.fillna({'Car': 0})

        if 'Bedroom2' in self.data.columns:
            self.data = self.data[self.data['Bedroom2'] < 10]
            

        return self.data

        
    def splitData(self, train_size=0.8, prediction_column='Price'):
        # From Quiz 1, split into training & test
        self.train_set = self.data.sample(frac=train_size)
        self.test_set = self.data.drop(self.train_set.index)

        self.train_X = self.train_set.drop(columns=prediction_column)
        self.train_Y = self.train_set['Price']

        self.test_X = self.test_set.drop(columns=prediction_column)
        self.test_Y = self.test_set['Price']

        print("training size", len(self.train_X))
        print("test size", len(self.test_X))

        return self.train_X, self.train_Y, self.test_X, self.test_Y
    
    def encodeCategoricalData(self, columns):
        # From Quiz 1, one-hot encoding

        # !!! still need to implement this
        pass 
        
    def reduceDataSize(self, n):
        self.data = self.data.head(n)
        return self.data
    
    def getData(self):
        return self.data