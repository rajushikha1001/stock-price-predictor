# src/preprocess.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_and_preprocess(path):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    scaler = MinMaxScaler()
    df['Close'] = scaler.fit_transform(df[['Close']])
    return df, scaler
