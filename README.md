# kaggle_ny_taxi
Kaggle Challenge: Predicting New York yellow cab prices 
Example 


model.add(Dense(output_dim = 384, init = 'glorot_normal', activation = 'relu', input_dim = X_train.shape[1]))
model.add(Dense(output_dim = 512, init = 'glorot_normal', activation = 'relu'))
model.add(Dense(output_dim = 1024, init = 'glorot_normal', activation = 'relu'))
model.add(Dense(output_dim = 768, init = 'glorot_normal', activation = 'relu'))
model.add(Dense(output_dim = 1, init = 'glorot_normal', activation = 'linear'))
