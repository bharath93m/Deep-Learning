import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import the dataset
training_set = pd.read_csv("D:/Udemy/Deep Learning/Recurrent_Neural_Networks/Google_Stock_Price_Train.csv")
training_set = training_set.iloc[:,1:2].values

#Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)

#get input and output
X_train = training_set[0:1257]
y_train = training_set[1:1258]

#reshape to match keras input requirements
X_train = np.reshape(X_train,(1257,1,1))

#import keras libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

regressor = Sequential()

regressor.add(LSTM(units=4,activation='sigmoid',input_shape=(None,1)))

regressor.add(Dense(units=1))

regressor.compile(optimizer='adam',loss='mean_squared_error')

regressor.fit(X_train,y_train,batch_size=32,epochs=200)

#making predictions
test_set = pd.read_csv("D:/Udemy/Deep Learning/Recurrent_Neural_Networks/Google_Stock_Price_Test.csv")
real_stock_price = training_set.iloc[:,1:2].values

inputs= real_stock_price
inputs = sc.transform(inputs)
inputs = np.reshape(inputs,(20,1,1))
predicted_stock_price = regressor.predict(inputs)

predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#visualizing the results
plt.plot(real_stock_price, color='red',label='Real Google Stock price')
plt.plot(predicted_stock_price, color='blue',label='Predicted Google Stock price')
plt.title('Google Stock price prediction')
plt.xlabel('Time')
plt.ylabel('Stock price')
plt.legend()
plt.show()
