import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from preprocess import load_and_preprocess

window = 5  # ✅ Define once and use consistently

df, scaler = load_and_preprocess('data/stock_data.csv')
data = df['Close'].values

if len(data) < window + 1:
    raise ValueError(
        f"Not enough data to train. Need at least {window + 1} rows, got {len(data)}.")

X, y = [], []
for i in range(window, len(data)):
    X.append(data[i-window:i])
    y.append(data[i])

X = np.array(X)
y = np.array(y)

if X.ndim != 2:
    raise ValueError(f"Expected X to be 2D, got shape {X.shape}")

X = X.reshape((X.shape[0], X.shape[1], 1))

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=10, batch_size=32)
model.save('src/model.keras')
print("✅ Model training complete and saved to src/model.keras")
