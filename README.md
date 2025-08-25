# Stock Price Prediction with Machine Learning (ARIMA & LSTM)

This project shows end-to-end **daily stock price forecasting** using:
- **ARIMA** (classical time-series modeling)
- **LSTM** (deep learning sequence modeling)

## Run
```bash
pip install -r requirements.txt
python src/train_stock_models.py --ticker AAPL --start 2018-01-01 --end 2025-08-20
