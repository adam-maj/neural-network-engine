# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


x_train = []
for row in X_train:
    x_train.append([item for item in row])

x_test = []
for row in X_test:
    x_test.append([item for item in row])

y_train = []
for item in Y_train:
    y_train.append([item])
    
y_test = []
for item in Y_test:
    y_test.append([item])

from NeuralNetworkEngine import NeuralNetwork

model = NeuralNetwork()
model.add_layer(11, input_layer = True)
model.add_layer(6, activation = 'sigmoid')
model.add_layer(6, activation = 'sigmoid')
model.add_layer(1, activation = 'sigmoid')
model.train(x_train, y_train, 0.3, 20)
model.predict(x_test, y_test)