#importing all the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset into the variable dataset
dataset=pd.read_csv('D:/Machine learning/Machine-Learning-A-Z-Codes-Datasets/Part 1 - Data Preprocessing/Section 2 -------------------- Part 1 - Data Preprocessing --------------------/Python/Data.csv')
x=dataset.iloc[:, :-1].values
y=dataset.iloc[:, -1].values

print(x)
print()
print(y)

#dealing with missing values
print(dataset.isnull().sum())
print()
#USING IMPUTER CLASS TO DEAL WITH MISSING VALUES
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit(x[:, 1:3])
x[:,1:3]=imputer.transform(x[:,1:3])

print(x)
print()

#dealing with the categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])], remainder='passthrough')
x=np.array(ct.fit_transform(x))
print(x)

from sklearn.preprocessing import LabelEncoder
lt = LabelEncoder()
y = lt.fit_transform(y)
print(y)

#dealing with splitting the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[:,3:] = sc.fit_transform(x_train[:,3:])
x_test[:,3:] = sc.transform(x_test[:,3:])

print(x_train)
print(x_test)
