import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Download historical data for Ethereum
eth = yf.Ticker("ETH-USD")
history = eth.history(period="max")

# Scale the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(history[['Close']])

# Split the data into training and test sets
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[0:train_size,:], scaled_data[train_size:len(scaled_data),:]

# Create a function to create a dataset with a rolling window
def create_dataset(dataset, window_size=1):
    data_X, data_Y = [], []
    for i in range(len(dataset) - window_size - 1):
        a = dataset[i:(i + window_size), 0]
        data_X.append(a)
        data_Y.append(dataset[i + window_size, 0])
    return np.array(data_X), np.array(data_Y)

# Use the function to create datasets for the training and test data
window_size = 5
train_X, train_Y = create_dataset(train_data, window_size)
test_X, test_Y = create_dataset(test_data, window_size)

# Reshape the input data to be 3D
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

# Create the LSTM model
model = Sequential()
model.add(LSTM(4, input_shape=(1, window_size)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Fit the model to the training data
model.fit(train_X, train_Y, epochs=100, batch_size=1, verbose=2)

# Use the model to make predictions on the test data
predictions = model.predict(test_X)

# Unscale the predictions
predictions = scaler.inverse_transform(predictions)

# Print the predictions
print(predictions)
