# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 00:35:40 2019

@author: Darshan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1]
y = dataset.iloc[:,1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)

# Fitting liner regression(Simple Linear Regression)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

test_value = regressor.predict(x_test)

diff_predict = y_test-test_value

# Comment out the below code for user interaction
#user_exp = float(input('Enter years of experience'))
#pred = regressor.predict(user_exp)

# Visualizing the training set of results 
plt.figure(figsize=(10,10))
plt.scatter(x_train,y_train,color='red')    
plt.plot(x_train, regressor.predict(x_train),color='blue')
plt.title('Salary vs Experience - Training')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(x_test,y_test,color='black')
plt.plot(x_train, regressor.predict(x_train),color='yellow')
plt.title('Salary vs Experience - Test')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()