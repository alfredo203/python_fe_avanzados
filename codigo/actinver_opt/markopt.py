# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 14:08:58 2024

@author: David
"""

import pandas as pd
import yfinance as yf
import numpy as np
import datetime

def get_stock_data(tickers, start, end):
    
    data = {}
    for ticker in tickers:
        data[ticker] = yf.download(ticker, start=start, end=end)["Close"]
    return pd.concat(data.values(), axis=1, keys=tickers)

def calculate_returns(stocks):
    
    returns = stocks / stocks.shift(1)
    log_returns = np.log(returns)
    return log_returns

def simulate_portfolios(log_returns, num_portfolios=500):
    
    num_stocks = log_returns.shape[1]
    weights = np.zeros((num_portfolios, num_stocks))
    expected_returns = np.zeros(num_portfolios)
    expected_volatilities = np.zeros(num_portfolios)
    sharpe_ratios = np.zeros(num_portfolios)

    mean_log_returns = log_returns.mean()
    cov_matrix = log_returns.cov()

    for k in range(num_portfolios):
        w = np.random.random(num_stocks)
        w /= np.sum(w)  # Normaliza los pesos para que sumen 1
        weights[k, :] = w
        
        expected_returns[k] = np.sum(mean_log_returns * w)
        expected_volatilities[k] = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        sharpe_ratios[k] = expected_returns[k] / expected_volatilities[k]

    return weights, expected_returns, expected_volatilities, sharpe_ratios

def calculate_opt():
    # Fechas
    start = datetime.datetime(2021, 1, 1)
    end = datetime.datetime(2021, 1, 22)

    # Datos de las acciones
    tickers = ["CEVA", "GOOGL", "TSLA", "ZOM"]
    stocks = get_stock_data(tickers, start, end)
    
    # Renombrar las columnas
    stocks.columns = ["CEVA", "GOOGLE", "TESLA", "ZOMEDICA"]
    print(stocks)
    
    # Calcular log returns
    log_returns = calculate_returns(stocks)
    
    # Simulación de portafolios
    weights = simulate_portfolios(log_returns)
    
    weights, expected_returns, expected_volatilities, sharpe_ratios = simulate_portfolios(log_returns)
    
    calculate_opt()
