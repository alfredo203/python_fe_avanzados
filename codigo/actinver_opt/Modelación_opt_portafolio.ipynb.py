# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 11:37:51 2024

@author: David
"""
import pandas as pd
import yfinance as yf
import datetime
import numpy as np
import matplotlib.pyplot as plt

# Fechas
start = datetime.datetime(2021, 1, 1)
end = datetime.datetime(2021, 1, 22)

# Datos de las acciones
ceva = yf.download("CEVA", start=start, end=end)
google = yf.download("GOOGL", start=start, end=end)
tesla = yf.download("TSLA", start=start, end=end)
zom = yf.download("ZOM", start=start, end=end)

# Concatenar las columnas
stocks = pd.concat([ceva["Close"], google["Close"], tesla["Close"], zom["Close"]], axis=1)

# Renombrar
stocks.columns = ["CEVA", "GOOGLE", "TESLA", "ZOMEDICA"]
 
print(stocks)

# Returns 
returns = stocks / stocks.shift(1)
print(returns)
 
#Logaritmos
logReturns = np.log(returns)

noDEPortafolios = 500
weight = np.zeros((noDEPortafolios, 4))
ReturnEsp = np.zeros(noDEPortafolios)
VolEsp = np.zeros(noDEPortafolios)
RadioSharpe = np.zeros(noDEPortafolios)

meanlogReturns = logReturns.mean()
Sigma = logReturns.cov()

for k in range(noDEPortafolios):
    w = np.array(np.random.random(4))
    w = w / np.sum(w)
    weight[k, :] = w
    
    ReturnEsp[k] = np.sum(meanlogReturns * w)
    
    VolEsp[k] = np.sqrt(np.dot(w.T, np.dot(Sigma, w)))
    
    # Sharpe Ratio
    RadioSharpe[k] = ReturnEsp[k] / VolEsp[k]

maxIndex = RadioSharpe.argmax()
weight[maxIndex, :]

# Grafica frontera eficiente
plt.figure(figsize=(10, 6))
plt.scatter(VolEsp, ReturnEsp, c=RadioSharpe, cmap="viridis", marker='o')
plt.xlabel("Riesgo Esperado (Volatilidad)")
plt.ylabel("Rendimiento Esperado")
plt.colorbar(label="Sharpe Ratio")

#El mejor radio Sharpe
plt.scatter(VolEsp[maxIndex], ReturnEsp[maxIndex], c="red", marker=".", s=200)
plt.show()
