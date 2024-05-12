# Contributions - everyone 

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

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
    
    def filterRentData(self):
        columns = self.data.columns
        # Posted On,BHK,Rent,Size,Floor,Area Type,Area Locality,City,Furnishing Status,Tenant Preferred,Bathroom,Point of Contact

        # remove posted on column
        if 'Posted On' in self.data.columns:
            self.data = self.data.drop(columns=['Posted On'])

        if 'Floor' in self.data.columns:
            self.data = self.data.drop(columns=['Floor'])
        
        if 'Area Type' in self.data.columns:
            self.data = self.data.drop(columns=['Area Type'])

        if 'Rent' in self.data.columns:
            self.data.dropna(subset=['Rent'], inplace=True)
            self.data = self.data[self.data['Rent'] > 0]

    
    def filterMelbourneData(self):
        
        # apply imuations
        columns = self.data.columns
        # self.data = self.applyImpuations(columns) # impute missing values

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

        if 'Latitude' in self.data.columns:
            self.data.dropna(subset=['Latitude'], inplace=True)

        if 'Longitude' in self.data.columns:
            self.data.dropna(subset=['Longitude'], inplace=True)
        
        if 'Type' in self.data.columns:
            self.data['Type'] = self.data['Type'].replace('t', 'h')

            self.data = self.data[self.data['Type'] == 'h']
            
            if 'Landsize' in self.data.columns and 'BuildingArea' in self.data.columns:
                self.data = self.data.dropna(subset=['Landsize', 'BuildingArea'])

                self.data['Size'] = self.data['Landsize']
                self.data.loc[self.data['Type'] == 'u', 'Size'] = self.data['BuildingArea']
                # self.data = self.data.drop(columns=['Landsize', 'BuildingArea'])

        if 'Car' in self.data.columns:
            self.data = self.data.fillna({'Car': 0})

        if 'Bedroom2' in self.data.columns:
            self.data = self.data[self.data['Bedroom2'] < 10]
        
        return self.data
    

    def encodeCategoricalData(self, columns):
        self.data = pd.get_dummies(self.data, columns=columns)
        return self.data

        
    def splitData(self, train_size=0.8, prediction_column='Price'):
        # From Quiz 1, split into training & test
        self.train_set = self.data.sample(frac=train_size)
        self.test_set = self.data.drop(self.train_set.index)

        self.train_X = self.train_set.drop(columns=prediction_column)
        self.train_Y = self.train_set[prediction_column]

        self.test_X = self.test_set.drop(columns=prediction_column)
        self.test_Y = self.test_set[prediction_column]

        print("training size", len(self.train_X))
        print("test size", len(self.test_X))

        return self.train_X, self.train_Y, self.test_X, self.test_Y

        
    def reduceDataSize(self, n):
        self.data = self.data.head(n)
        return self.data
    
    def getData(self):
        return self.data

    def getAverage(self, column):
        try:
            return self.data[column].mean()
        except KeyError:
            return None
        
    def remove_outliers(self, columns, plot=False):
        # Filter columns to include only numeric data types
        numeric_columns = [col for col in columns if pd.api.types.is_numeric_dtype(self.data[col])]

        number_of_rows_before = self.data.shape[0]

        for column in numeric_columns:
            # Calculate IQR for each column
            Q1 = self.data[column].quantile(0.25)
            Q3 = self.data[column].quantile(0.75)
            IQR = Q3 - Q1

            bound_factor = 0.5

            lower_bound = Q1 - bound_factor * IQR
            upper_bound = Q3 + bound_factor * IQR

            # Remove rows where the column value is outside the IQR bounds
            self.data = self.data[(self.data[column] >= lower_bound) & (self.data[column] <= upper_bound)]

            # Optionally plot the cleaned data
            if plot:
                plt.figure(figsize=(4, 3))
                ax = sns.boxplot(
                    x=self.data[column],
                    boxprops={'facecolor': 'lightblue', 'alpha': 0.6},
                    flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 8, 'alpha': 0.9},
                    whiskerprops={'color': 'black'},
                    capprops={'color': 'black'}
                )
                ax.set_title(f'Boxplot after Outlier Removal: {column}')
                ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                plt.show()

        number_of_rows_after = self.data.shape[0]
        print(f"Number of rows removed: {number_of_rows_before - number_of_rows_after}")
        print(f"Number of rows remaining: {number_of_rows_after}")

        return self.data
    
    # currently not working
    def applyImpuations(self, columns):
        imputer = SimpleImputer(strategy='mean')

        # only apply to columns with numbers 
        numeric_columns = [col for col in columns if pd.api.types.is_numeric_dtype(self.data[col])]
        self.data[numeric_columns] = imputer.fit_transform(self.data[numeric_columns])
        return self.data
    

    def calculateCorelation(self, columns, plot=True):

        # columns - list of columns to calculate correlation
        self.correlation_matrix = self.data[columns].corr()
        if plot: 
            plt.figure(figsize=(6, 5))
            sns.heatmap(self.correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
            plt.title('Correlation Matrix Heatmap')
            plt.show()
        return self.correlation_matrix
