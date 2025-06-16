# STOCK-PREDICTION-SOFTWARE
///PYTHON BASED PREDICTOR USING MACHINE LEARNING///
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

def load_data(filename):
    df = pd.read_csv(filename)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df[['Close']]


def create_sequences(data, window_size=60):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


if __name__ == "__main__":
    
    TICKER = 'AAPL'
    TRAIN_SPLIT = 0.8
    WINDOW_SIZE = 60
    
    
    dataset = load_data(f'{TICKER}.csv').values
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    
    
    train_data = scaled_data[0:int(len(scaled_data)*TRAIN_SPLIT), :]
    X_train, y_train = create_sequences(train_data, WINDOW_SIZE)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    
    model = build_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, batch_size=64, epochs=50)
    
    
    test_data = scaled_data[len(train_data)-WINDOW_SIZE:, :]
    X_test = []
    X_test.append(test_data)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    
actual_prices = dataset[len(train_data):]
print("Predicted Prices vs Actual Prices:")
for pred, actual in zip(predictions.flatten(), actual_prices.flatten()):
    print(f"Predicted: ${pred:.2f}, Actual: ${actual:.2f}")

plt.plot(actual_prices, color='blue', label='Actual Stock Price')
plt.plot(predictions, color='red', label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

