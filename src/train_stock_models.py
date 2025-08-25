import argparse, os, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

from utils import train_val_test_split_series, make_sliding_windows, rmse, mape

def fetch_data(ticker, start, end, price_field='Close'):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    series = df[price_field].dropna().astype(float)
    series.name = f"{ticker}_{price_field}"
    return series

def grid_search_arima(train, val):
    best_cfg, best_rmse = None, np.inf
    for p in range(0,3):
        for d in range(0,2):
            for q in range(0,3):
                try:
                    model = ARIMA(train, order=(p,d,q)).fit()
                    fc = model.forecast(steps=len(val))
                    err = rmse(val.values, fc.values)
                    if err < best_rmse:
                        best_rmse = err
                        best_cfg = (p,d,q)
                except:
                    continue
    return best_cfg or (0,1,0)

def fit_predict_arima(train, val, test):
    cfg = grid_search_arima(train, val)
    model = ARIMA(pd.concat([train,val]), order=cfg).fit()
    return model.forecast(steps=len(test)).values

def fit_predict_lstm(train, val, test, seq_len=60, epochs=20):
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.values.reshape(-1,1))
    val_scaled = scaler.transform(val.values.reshape(-1,1))
    test_scaled = scaler.transform(test.values.reshape(-1,1))

    X_train, y_train = make_sliding_windows(train_scaled.flatten(), seq_len)
    tv_scaled = np.concatenate([train_scaled.flatten(), val_scaled.flatten()])
    X_val, y_val = make_sliding_windows(tv_scaled, seq_len)
    X_val, y_val = X_val[-len(val):], y_val[-len(val):]

    model = Sequential([LSTM(64, input_shape=(seq_len,1)), Dense(1)])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, validation_data=(X_val,y_val),
              epochs=epochs, batch_size=32, verbose=0,
              callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])
    
    tvt_scaled = np.concatenate([train_scaled.flatten(), val_scaled.flatten(), test_scaled.flatten()])
    X_all, _ = make_sliding_windows(tvt_scaled, seq_len)
    X_test = X_all[-len(test):]
    y_pred = scaler.inverse_transform(model.predict(X_test, verbose=0)).flatten()
    return y_pred, model

def plot_predictions(train, val, test, preds, title, outpath):
    plt.figure(figsize=(10,5))
    plt.plot(train.index, train.values, label='Train')
    plt.plot(val.index, val.values, label='Val')
    plt.plot(test.index, test.values, label='Test')
    plt.plot(test.index, preds, label='Forecast')
    plt.title(title); plt.legend(); plt.tight_layout()
    plt.savefig(outpath); plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', default='AAPL')
    parser.add_argument('--start', default='2018-01-01')
    parser.add_argument('--end', default=None)
    parser.add_argument('--price_field', default='Close')
    parser.add_argument('--seq_len', type=int, default=60)
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)

    series = fetch_data(args.ticker, args.start, args.end, args.price_field)
    train,val,test = train_val_test_split_series(series)

    # ARIMA
    arima_preds = fit_predict_arima(train,val,test)
    arima_rmse, arima_mape = rmse(test.values, arima_preds), mape(test.values, arima_preds)
    plot_predictions(train,val,test,arima_preds,"ARIMA Forecast","outputs/arima_predictions.png")

    # LSTM
    lstm_preds, model = fit_predict_lstm(train,val,test,args.seq_len,args.epochs)
    lstm_rmse, lstm_mape = rmse(test.values, lstm_preds), mape(test.values, lstm_preds)
    plot_predictions(train,val,test,lstm_preds,"LSTM Forecast","outputs/lstm_predictions.png")
    model.save("outputs/lstm_model.keras")

    # Metrics
    metrics = {
        "ARIMA":{"RMSE":arima_rmse,"MAPE":arima_mape},
        "LSTM":{"RMSE":lstm_rmse,"MAPE":lstm_mape}
    }
    print(metrics)
    import json; json.dump(metrics, open("outputs/metrics.json","w"), indent=2)

if __name__ == "__main__":
    main()
