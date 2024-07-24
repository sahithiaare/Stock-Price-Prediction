import os
import numpy as np import pandas as pd import matplotlib.pyplot as plt %matplotlib inline
dataset_train=pd.read_csv"Google_Stock_Price_Train.csv")
dataset_train.head ()
training_set=dataset_train.iloc[:,1:2].values
print (training_set)
print (training_set. shape)
from sklearn.preprocessing import MinMaxScaler scaler=MinMaxScaler (feature_range=(0,1))
scaled_training_set=scaler.fit_transform(training_set)
scaled_training_set
x_train=[]
y_train=[]
for i in range (60,1258):
x_train.append(scaled_training_set[i-60:1,0])
y_train. append (scaled_training_set(i,0])
x=np. array(x_train)
yenp.array (y_train)
print (x. shape) print (y.shape)
x=np.reshape(x, (x.shape[0],x.shape[1],1))
x. shape
pip install keras
pip install TensorFlow
from keras.models import Sequential from
keras.layers import LSTM
from
keras.layers import Dense from keras.layers import Dropout from sklearn.metrics import mean_squared_error from sklearn.ensemble import RandomForestRegressor
regressor=Sequential()
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(x.shape [1),1)))
regressor.add (Dropout(0.2))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add (Dropout(0.2))
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50))
regressor.add (Dropout(0.2)) regressor.add(Dense(units=1))
regressor.compile(optimizer='adam', loss= 'mean_squared_error')
regressor.fit(x,y,epochs=100,batch_size=32)
dataset_test=pd.read_csv("Google_Stock_Price_Train.csv")
actual_stock_price=dataset_test. iloc[:,1:2].values
dataset_total=pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)
inputs=dataset_total [len(dataset_total)-len(dataset_test)-60:].values
inputs=inputs.reshape(-1,1)
inputs=scaler.transform(inputs)
for i in range (60,80):
x_test.append(inputs[i-60:1,0])
x_test=p.array(x_test)
x_test=np.reshape(x_test, (x_test.shape[0],_test.shape[1],1))
predicted_stock_price=regressor.predict(x_test)
predicted_stock_price=scaler.inverse_transform(predicted_stock_price)
plt.plot(actual_stock_price,color='red',label="Actual google stock price")
plt.plot(predicted_stock_price,color='blue' ,label="predicted google stock price")
plt.title('google stock price prediction') plt.xlabel('time')
plt.ylabel('google stock price')
plt. legend)
