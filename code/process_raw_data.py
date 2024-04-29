# get data from dataset folder

import os
import pandas as pd

def get_data():
    data = pd.DataFrame()

    filepath = 'dataset/Melbourne_housing_FULL.csv'
    
    if os.path.exists(filepath):
        data = pd.read_csv(filepath)
    else:
        print("File not found")

    return data

data = get_data()