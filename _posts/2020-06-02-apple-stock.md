### Description: 

This program uses an Artificial Recurrent Neural network called Long Short Term Memory (LSTM) to predict the closing stock price of a corporation (Aplle Inc.) using the past 60 days stock price. Stock data is provided from Yahoo Finance. The target future in this prediction analysis is 'Close' price instead of 'Adjusted Close' price.

#### Get the data

First off all, the dependencies are installed.


```python
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:45% !important; }</style>"))
```


<style>.container { width:45% !important; }</style>



```python
import numpy as np
import math
import pandas_datareader as web
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
```




After installing the dependencies, we shall move to acquire the data from 
website through package pandas_datareader. In this analysis, data from June 
1st, 2012 to June 1st, 2020 are requested to formulate the initial dataframe.


```python
df = web.DataReader('AAPL', data_source ='yahoo', start ='2012-06-01', end ='2020-06-01')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Adj Close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2012-06-01</th>
      <td>81.807144</td>
      <td>80.074287</td>
      <td>81.308571</td>
      <td>80.141426</td>
      <td>130246900.0</td>
      <td>69.378235</td>
    </tr>
    <tr>
      <th>2012-06-04</th>
      <td>81.071426</td>
      <td>78.357140</td>
      <td>80.214287</td>
      <td>80.612854</td>
      <td>139248900.0</td>
      <td>69.786331</td>
    </tr>
    <tr>
      <th>2012-06-05</th>
      <td>80.924286</td>
      <td>79.761429</td>
      <td>80.181427</td>
      <td>80.404289</td>
      <td>97053600.0</td>
      <td>69.605789</td>
    </tr>
    <tr>
      <th>2012-06-06</th>
      <td>81.978569</td>
      <td>80.785713</td>
      <td>81.110001</td>
      <td>81.637146</td>
      <td>100363900.0</td>
      <td>70.673080</td>
    </tr>
    <tr>
      <th>2012-06-07</th>
      <td>82.474289</td>
      <td>81.500000</td>
      <td>82.470001</td>
      <td>81.674286</td>
      <td>94941700.0</td>
      <td>70.705200</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Adj Close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-05-26</th>
      <td>324.239990</td>
      <td>316.500000</td>
      <td>323.500000</td>
      <td>316.730011</td>
      <td>31380500.0</td>
      <td>316.730011</td>
    </tr>
    <tr>
      <th>2020-05-27</th>
      <td>318.709991</td>
      <td>313.089996</td>
      <td>316.140015</td>
      <td>318.109985</td>
      <td>28236300.0</td>
      <td>318.109985</td>
    </tr>
    <tr>
      <th>2020-05-28</th>
      <td>323.440002</td>
      <td>315.630005</td>
      <td>316.769989</td>
      <td>318.250000</td>
      <td>33390200.0</td>
      <td>318.250000</td>
    </tr>
    <tr>
      <th>2020-05-29</th>
      <td>321.149994</td>
      <td>316.470001</td>
      <td>319.250000</td>
      <td>317.940002</td>
      <td>38399500.0</td>
      <td>317.940002</td>
    </tr>
    <tr>
      <th>2020-06-01</th>
      <td>322.350006</td>
      <td>317.209991</td>
      <td>317.750000</td>
      <td>321.850006</td>
      <td>20197800.0</td>
      <td>321.850006</td>
    </tr>
  </tbody>
</table>
</div>



Get the number of rows and columns in the data set:


```python
df.shape
```




    (2012, 6)



Use plt to visualize the trend of closing price in the past 8 years:


```python
plt.figure(figsize=(6,4))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.show()
```


![alt](/assets/images/output_10_0.png)


If you are the lucky guy who bought the stock of Apple around mid of 2013, where the price was roughly about $60, today your asset is going to be worth 321 ish. 

#### Organize the data

In the second step the initial data set needs to be organized into machine learning data sets which can be used to fit the machine learning models.


```python
#Create a new dataframe with only the 'Close' column
data = df.filter(['Close'])
#Convert the dataframe to a numpy array
dataset = data.values
#Get the number of rows to train the model on
training_data_len = math.ceil(len(dataset) * .8)

training_data_len
```




    1610



Scale the data into [0,1]


```python
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

scaled_data
```




    array([[0.08972191],
           [0.09145887],
           [0.09069042],
           ...,
           [0.96702402],
           [0.96588184],
           [0.9802881 ]])




```python
#Create the  training data set
#Create the scaled training data set
train_data = scaled_data[0:training_data_len, :]
#Split the data into x_train and y_train data sets
x_train = []
y_train = []

print(train_data)
print(len(train_data))
```

    [[0.08972191]
     [0.09145887]
     [0.09069042]
     ...
     [0.60248329]
     [0.60742046]
     [0.61508415]]
    1610



```python
# the past 60 values
for i in range(60, len(train_data)):
  x_train.append(train_data[i-60:i])
  y_train.append(train_data[i,0])
  if i<= 60:
    print(x_train)
    print(y_train)
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-30-ebbb1aff9196> in <module>
          1 # the past 60 values
          2 for i in range(60, len(train_data)):
    ----> 3   x_train.append(train_data[i-60:i])
          4   y_train.append(train_data[i,0])
          5   if i<= 60:


    AttributeError: 'numpy.ndarray' object has no attribute 'append'



```python
#Convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)
```

Reshape the data:

LSTM model expects the data shape to be 3 dimensional in the form of 1) number of samples, 2) number of steps, and 3) number of feature.

Number of samples means how many rows inside the set. Numebr of steps refers to the number of columns. Number of feature refers to 1 in this case because we are going to predict the 'Close' price only.


```python
x_train.shape
```




    (1550, 60, 1)



#### Build the LSTM model and fit the data into the model


```python
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
```


```python
#Compile the model
model.compile(optimizer = 'adam', loss='mean_squared_error')
```


```python
#Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)
```

    Epoch 1/1
    1550/1550 [==============================] - 87s 56ms/step - loss: 6.2108e-04





    <keras.callbacks.callbacks.History at 0x13d7def40>



Create the test data set:


```python
#create a new array containing scaled values from index 1550 to 2012
test_data = scaled_data[training_data_len - 60: , :]
#Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len: , :]

for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i, 0])
```


```python
#Convert the data to a numpy array
x_test = np.array(x_test)
```


```python
#reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
x_test.shape
```




    (402, 60, 1)



Get the predicted price values:


```python
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
```

Calculate the root mean squared error (RMSE):


```python
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
rmse
```




    9.41572357726652



Visualize the predicted values and the actual values:


```python
#plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
#visualize
plt.figure(figsize=(6,4))
plt.title('Model Prediction')
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Close Price USD ($)', fontsize = 18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Valid', 'Predictions'], loc='lower right')
plt.show
```

    <ipython-input-31-f16700053db2>:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      valid['Predictions'] = predictions





    <function matplotlib.pyplot.show(*args, **kw)>




![png](/assets/images/output_34_2.png)


Basically from the figure above we can see the trend of predicted stock price (yellow) follows the actual trend (red).


```python
#show the valid and the predicted prices
valid
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Close</th>
      <th>Predictions</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-10-24</th>
      <td>215.089996</td>
      <td>225.873962</td>
    </tr>
    <tr>
      <th>2018-10-25</th>
      <td>219.800003</td>
      <td>225.406982</td>
    </tr>
    <tr>
      <th>2018-10-26</th>
      <td>216.300003</td>
      <td>225.261795</td>
    </tr>
    <tr>
      <th>2018-10-29</th>
      <td>212.240005</td>
      <td>224.814850</td>
    </tr>
    <tr>
      <th>2018-10-30</th>
      <td>213.300003</td>
      <td>223.819748</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2020-05-26</th>
      <td>316.730011</td>
      <td>314.553467</td>
    </tr>
    <tr>
      <th>2020-05-27</th>
      <td>318.109985</td>
      <td>315.355865</td>
    </tr>
    <tr>
      <th>2020-05-28</th>
      <td>318.250000</td>
      <td>316.063568</td>
    </tr>
    <tr>
      <th>2020-05-29</th>
      <td>317.940002</td>
      <td>316.641693</td>
    </tr>
    <tr>
      <th>2020-06-01</th>
      <td>321.850006</td>
      <td>317.054230</td>
    </tr>
  </tbody>
</table>
<p>402 rows × 2 columns</p>
</div>




```python
#To predict the price of 2020-6-2
#Get the qoute
apple_qoute = web.DataReader('AAPL', data_source = 'yahoo', start='2012-06-01', 
                             end = '2020-06-01')
#Create a new dataframe
new_df = apple_qoute.filter(['Close'])
#get the last 60 day closing values and convert the dataframe to an array
last_60_days = new_df[-60:].values
#scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)
#create an empty list
X_test = []
#Append the past 60 days
X_test.append(last_60_days_scaled)
#Convert the X_test data set to a numpy array
X_test = np.array(X_test)
#Reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#Get the predicted scaled price
pred_price = model.predict(X_test)
#undo the scaling
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)
```

    [[317.82578]]



```python
#Get the qoute
apple_qoute2 = web.DataReader('AAPL', data_source = 'yahoo', start='2020-06-02', 
                             end = '2020-06-02')
print(apple_qoute2['Close'])
```

    Date
    2020-06-02    323.339996
    Name: Close, dtype: float64


The model prediction gives the predicted Apple stock price on second of June, 2020 is $307.87

While the actual Apple stock price on that day is $323.34 acquired from Yahoo Finance platform.

In this analysis, my goal is to familiarise myself with how to acquire historical stock price data from a specific company, which is Apple in this case. Also, the application of machine learning model through Python language is quite interesting to learn here. However, it is not wise to predict the stock price through this simple program since there are numerous ups and downs in each trading day, which is hard to grasp the pattern so to perform the prediction. Overall, it is nice to play with and interesting to visualize the analysis.
