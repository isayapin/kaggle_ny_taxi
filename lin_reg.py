#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun Sep  2 17:16:15 2018
@author: isayapin
"""

#Simple Linear Regression

# Importing the libraries
import numpy as np
import pandas as pd
from data_processing import data_processing

X_train, X_test, y_train, y_test, sc_y = data_processing(1000000)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Evaluating performance
y_pred = regressor.predict(X_test)
y_pred = sc_y.inverse_transform(y_pred)

from sklearn.metrics import mean_squared_error
print(np.sqrt(mean_squared_error(y_pred, y_test)))


"""
RMSE = 4.343727274333207
"""