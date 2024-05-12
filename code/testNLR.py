# loading in a sample dataset and testing the NLRegression class

from sklearn.datasets import load_boston
from sklearn.datasets import make_regression
import pandas as pd
from sklearn.model_selection import train_test_split
from NLRegression import NLRegression
import matplotlib.pyplot as plt

# Load data and convert to DataFrame
# data = make_regression(n_samples=400, n_features=3, noise=4)
data = load_boston()
print(data.keys())

# plot each feature against price
for i in range(3):
    plt.scatter(data.data[:, i], data.target)
    plt.xlabel(data.feature_names[i])
    plt.ylabel("Price")
    plt.show()


# only use the first 5 features for training
# data.data = data.data[:, :7]
# data.feature_names = data.feature_names[:7]

# df = pd.DataFrame(data[0], columns=["Feature" + str(i) for i in range(1, 4)])

df = pd.DataFrame(data.data, columns=data.feature_names)
target = pd.Series(data.target, name="Price")
# target = pd.Series(data[1], name="Price")

# Split into train and test
train_X, test_X, train_Y, test_Y = train_test_split(df, target, test_size=0.2, random_state=42)

features = list(df.columns)
print("Features:", features)

print("Training data shape:", train_X.shape)

model = NLRegression(train_X, train_Y, test_X, test_Y, features)

model.train()
model.cross_validate()
model.plot()

# for i in range(5):
#     pred, bounds = model.predictActual(test_X.iloc[i])
#     print("prediction:  ", pred[0])
#     print("bounds:      ", bounds[0], bounds[1])
#     print("actual:      ", test_Y.iloc[i])
