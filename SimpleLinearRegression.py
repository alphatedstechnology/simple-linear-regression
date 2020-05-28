# Simple Linear Regression
#
#   **y = b0 + b1*x1**
#
#- y = dependent variable
#- x = independent variable 
#- b1 = coefficient ( a unit change in x1 and its effect in y )-slope of the line
#- b0 = constant ( point where line crosses the vertical axes ) when x1 is 0
#
#  **Finds a line with minimum sum of error**
#
#    SUM(y - y')^2

#DATA PREPROCESSING

#IMPORTING THE DATASETS

#numpy is a mathematicaltool
import numpy as np
#matplotlib for visualization
import matplotlib.pyplot as plt
#panads for managing dataset
import pandas as pd

#< --- IMPORTING THE DATASET ---->
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,0].values
y = dataset.iloc[:,1].values


#splitting data into train, test, split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/3,random_state=0)

X_train= X_train.reshape(-1, 1)
y_train= y_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

#training the SIMPLE LINEAR REGRESSION model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#predicting

y_pred = regressor.predict(X_test)

#plotting the values
plt.scatter(X_train, y_train)
plt.plot(X_test, y_pred)