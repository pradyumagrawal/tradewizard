from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pickle

app = Flask(__name__)

# Load trained inflation model
inflation_model = tf.keras.models.load_model('inflation_model.h5')
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict_investment():
    data = request.json
    symbol = data.get('symbol', 'AAPL')
    
    # Fetch stock data
    df = yf.download(symbol, period='1y', interval='1d')
    df['Return'] = df['Close'].pct_change()
    latest_return = df['Return'].iloc[-1]

    # Predict future inflation
    gdp = data['gdp']
    unemployment = data['unemployment']
    prediction_input = scaler.transform([[gdp, unemployment]])
    inflation_pred = inflation_model.predict(prediction_input)[0][0]

    # Decision rule
    if inflation_pred < 2 and latest_return > 0:
        recommendation = "Invest: Stable market and low inflation expected."
    elif inflation_pred > 4:
        recommendation = "Withdraw: Inflation and market risk increasing."
    else:
        recommendation = "Hold: Market stable but uncertain inflation."

    return jsonify({
        "inflation_prediction": inflation_pred,
        "investment_decision": recommendation
    })

if __name__ == '__main__':
    app.run(debug=True)
