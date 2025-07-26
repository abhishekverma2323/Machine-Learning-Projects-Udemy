import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('D:/Machine learning/Machine-Learning-A-Z-Codes-Datasets/Part 2 - Regression/Section 4 - Simple Linear Regression/Python/Salary_Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_train)

plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary vs Year of experience ( Train-Set) ')
plt.xlabel('Experince (in years) ')
plt.ylabel('Salary')
plt.show()

plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary vs Year of experience ( Test-Set) ')
plt.xlabel('Experince (in years) ')
plt.ylabel('Salary')
plt.show()
print(regressor.predict([[12]]))