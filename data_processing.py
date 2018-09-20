#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Data preprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
    
def data_processing(n_rows):
    # Importing the libraries
    # Setting types to optimize memory usage
    types = {'fare_amount': 'float32',
             'pickup_longitude': 'float32',
             'pickup_latitude': 'float32',
             'dropoff_longitude': 'float32',
             'dropoff_latitude': 'float32',
             'passenger_count': 'uint8'}
    
    # Importing the dataset
    df = pd.read_csv('train.csv', nrows= n_rows, dtype=types)
    df = df.replace(0, np.nan)
    df = df.dropna(how='any',axis=0)
    
    #Visualising the fare distribution
    #binwidth = 20
    #bins = np.arange(min(df.fare_amount), max(df.fare_amount) + binwidth, binwidth)
    #plt.hist(df.fare_amount, bins = bins, histtype='bar', rwidth=0.8)
    """
    plt.hist(df.fare_amount, bins =100, range = [min(df.fare_amount), 100], histtype='bar', rwidth=0.8)
    plt.xlabel('fares')
    plt.ylabel('y')
    plt.show()
    """
    #Some of the prices are negative or too large
    df = df[(df.fare_amount > 0) & (df.fare_amount < 40)]
    description = df.describe()
    
    #Anomalies in the log/lat values
    df=df[(-76 <= df['pickup_longitude']) & (df['pickup_longitude'] <= -72)]
    df=df[(-76 <= df['dropoff_longitude']) & (df['dropoff_longitude'] <= -72)]
    df=df[(38 <= df['pickup_latitude']) & (df['pickup_latitude'] <= 42)]
    df=df[(38 <= df['dropoff_latitude']) & (df['dropoff_latitude'] <= 42)]
    df = df[(df['dropoff_longitude'] != df['pickup_longitude'])]
    df = df[(df['dropoff_latitude'] != df['pickup_latitude'])]
    description = df.describe()
    
    def weekday_evening(row):
        if ((row['hour'] <= 20) and (row['hour'] >= 16)) and (row['weekday'] < 5):
            return 1
        else:
            return 0
        
    #Getting date data
    df['pickup_datetime'] =  pd.to_datetime(df['pickup_datetime'], format='%Y-%m-%d %H:%M:%S %Z')
    df['month'] = df['pickup_datetime'].apply(lambda x: x.month)
    df['hour'] = df['pickup_datetime'].apply(lambda x: x.hour)
    df['weekday'] = df['pickup_datetime'].apply(lambda x: x.weekday())
    df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)
    df['is_late_night'] = df['hour'].apply(lambda x: 1 if x <= 6 or x >= 20 else 0)
    df['is_weekday_evening'] = df.apply (lambda x: weekday_evening(x), axis=1)
    
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()
    df['manhattan_distance'] = df['abs_diff_longitude'] + df['abs_diff_latitude']
    df['squared_long'] = np.power(df['abs_diff_longitude'],2)
    df['squared_lat'] = np.power(df['abs_diff_latitude'],2)
    df['euclid_distance'] = np.sqrt(df['squared_long'] + df['squared_lat'])
    
    features = ['pickup_longitude', 
                'pickup_latitude', 
                'dropoff_longitude', 
                'dropoff_latitude', 
                'passenger_count', 
                'month', 
                'is_weekend', 
                'is_late_night',
                'is_weekday_evening',
                'manhattan_distance', 
                'euclid_distance']
    
    X = df[features]
    y = df['fare_amount'].values
    y = y.reshape(-1, 1)
    
    # Encoding categorical data
    from sklearn.preprocessing import OneHotEncoder
    onehotencoder = OneHotEncoder(categorical_features = [features.index('month')])
    X = onehotencoder.fit_transform(X).toarray()
    X = X[:, 1:]
    
    # Splitting the dataset into the Training set and Test set
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    
    # Feature Scaling
    """
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    sc_X = MinMaxScaler()
    #scale_index = [11, 12, 13, 14, 19, 20, 21, 22]
    scale_index = [11, 12, 13, 14, 19, 20]
    X_train[:, scale_index] = sc_X.fit_transform(X_train[:, scale_index])
    X_test[:, scale_index] = sc_X.fit_transform(X_test[:, scale_index])
    
    sc_y = StandardScaler()
    y_train = sc_y.fit_transform(y_train)
    """
    
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    sc_y = StandardScaler()
    y_train = sc_y.fit_transform(y_train)
    
    return X_train, X_test, y_train, y_test, sc_y