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
end = datetime.datetime(2024, 1, 22)
a = yf.download("OXY1.MX", start=start, end=end)
b = yf.download("AAPL", start=start, end=end)
c = yf.download("TMO", start=start, end=end)
d = yf.download("QCOM", start=start, end=end)
e = yf.download("BAC", start=start, end=end)
f = yf.download("WFC", start=start, end=end)
g = yf.download("PFE", start=start, end=end)
h = yf.download("MEGACPO.MX", start=start, end=end)
i = yf.download("SKLZ", start=start, end=end)
j = yf.download("CVX", start=start, end=end)
k = yf.download("TGT", start=start, end=end)
l = yf.download("NKE", start=start, end=end)
m = yf.download("FANG", start=start, end=end)
n = yf.download("SBUX", start=start, end=end)
# Datos de las accionesart=start, end=end)

# Concatenar las columnas
stocks = pd.concat([a["Close"]
                    , b["Close"], c["Close"], d["Close"], e["Close"], f["Close"]
                    , g["Close"], h["Close"], i["Close"], j["Close"], k["Close"]
                    , l["Close"], m["Close"]
                    , n["Close"]], axis=1)

# Renombrar
stocks.columns = ["OXY1.MX", "AAPL", "TMO", "QCOM", "BAC", "WFC", "PFE", "MEGACPO.MX", "SKLZ", "CVX", "TGT", "NKE", "FANG", "SBUX"]

 
print(stocks)

# Returns 
returns = stocks / stocks.shift(1)
print(returns)
 
#Logaritmos
logReturns = np.log(returns)

noDEPortafolios = 10000
weight = np.zeros((noDEPortafolios, 14))
ReturnEsp = np.zeros(noDEPortafolios)
VolEsp = np.zeros(noDEPortafolios)
RadioSharpe = np.zeros(noDEPortafolios)

meanlogReturns = logReturns.mean()
Sigma = logReturns.cov()

for k in range(noDEPortafolios):
    w = np.array(np.random.random(14))
    w = w / np.sum(w)
    weight[k, :] = w
    
    ReturnEsp[k] = np.sum(meanlogReturns * w)
    
    VolEsp[k] = np.sqrt(np.dot(w.T, np.dot(Sigma, w)))
    
    # Sharpe Ratio
    RadioSharpe[k] = ReturnEsp[k] / VolEsp[k]

maxIndex = RadioSharpe.argmax()
weight[maxIndex, :]
pesos = weight[maxIndex, :]

# Grafica frontera eficiente
plt.figure(figsize=(10, 6))
plt.scatter(VolEsp, ReturnEsp, c=RadioSharpe, cmap="viridis", marker='o')
plt.xlabel("Riesgo Esperado (Volatilidad)")
plt.ylabel("Rendimiento Esperado")
plt.colorbar(label="Sharpe Ratio")

#El mejor radio Sharpe
plt.scatter(VolEsp[maxIndex], ReturnEsp[maxIndex], c="red", marker=".", s=200)
plt.show()
