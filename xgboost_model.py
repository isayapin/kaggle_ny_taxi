#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 15:35:59 2018

@author: isayapin
"""

#XGboost

# Importing the libraries
import numpy as np
import pandas as pd
from data_processing import data_processing

X_train, X_test, y_train, y_test, sc_y = data_processing(2000000)

from xgboost import XGBRegressor
model = XGBRegressor()

model.fit(X_train, y_train, verbose=False)

y_pred = model.predict(X_test)
y_pred = sc_y.inverse_transform(y_pred)

from sklearn.metrics import mean_squared_error
print(np.sqrt(mean_squared_error(y_pred, y_test)))

"""
RMSE = 2.7817755
"""
