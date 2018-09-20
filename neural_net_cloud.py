#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#DNNs on the cloud

# Importing the libraries
import numpy as np
import pandas as pd
from data_processing import data_processing
import keras
from keras import backend
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization

X_train, X_test, y_train, y_test, sc_y = data_processing(3000000)

# kernel_initializerialising the ANN
model = Sequential()

# Adding the input layer and the first hidden layer

model.add(Dense(units = 384, kernel_initializer = 'glorot_normal', activation = 'relu', input_dim = X_train.shape[1]))
model.add(Dense(units = 512, kernel_initializer = 'glorot_normal', activation = 'relu'))
model.add(Dense(units = 1024, kernel_initializer = 'glorot_normal', activation = 'relu'))
model.add(Dense(units = 1024, kernel_initializer = 'glorot_normal', activation = 'relu'))
model.add(Dense(units = 768, kernel_initializer = 'glorot_normal', activation = 'relu'))
#model.add(Dense(units = 512, kernel_initializer = 'glorot_normal', activation = 'relu'))
model.add(Dense(units = 1, kernel_initializer = 'glorot_normal', activation = 'linear'))

#Defining RMSE
def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

# Compiling and fitting the ANN
model.compile(optimizer = 'sgd', loss = 'mse', metrics = [rmse])
model.fit(X_train, y_train, batch_size = 50000, nb_epoch = 2000)

# Part 3 - Making the predictions and evaluating the model
y_pred = model.predict(X_test)
y_pred = sc_y.inverse_transform(y_pred)

from sklearn.metrics import mean_squared_error
print(np.sqrt(mean_squared_error(y_pred, y_test)))

model.save('neural_net_2.h5')