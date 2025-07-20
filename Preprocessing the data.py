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