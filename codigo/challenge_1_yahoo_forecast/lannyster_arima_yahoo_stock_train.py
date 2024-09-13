# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 13:43:53 2024

@author: edson
"""

# Importar paqueterias
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf

# Cargar los datos
path = "C:/Users/edson/OneDrive/Documents/Equpo Python/yahoo_stock_train.csv"
df = pd.read_csv(path)[['Date','Close']]

df_ajust = df.tail(120)# Seleccionar los últimos 120 datos

# Convertir la columna 'Date' en tipo datetime
df_ajust['Date'] = pd.to_datetime(df_ajust['Date'])

# Establecer la columna 'Date' como índice
df_ajust.set_index('Date', inplace=True)

df.head()

plt.figure(figsize=(12,6))
plt.plot(df_ajust, color='r')
plt.title('Evolución del precio en los últimos 90 días')

# Revisar si hay valores NaN o infinitos en los datos
print(df_ajust.isna().sum())  # Ver cuántos valores NaN hay en cada columna
print(np.isinf(df_ajust).sum())  # Ver cuántos valores inf hay en cada columna

# Eliminar o rellenar los valores faltantes (NaN) 
df_ajust = df_ajust.replace([np.inf, -np.inf], np.nan) # Reemplazar infinitos por NaN
df_ajust = df_ajust.dropna()  # Eliminar filas con NaN

# Verificar estacionariedad con la prueba de Dickey-Fuller
result = adfuller(df_ajust['Close'])
print(f"Estadístico de Dickey-Fuller: {result[0]}")
print(f"Valor p: {result[1]}")

#Segunda prueba con los datos diferenciados 1 vez
result = adfuller(df_ajust['Close'].diff().dropna())
print(f"Estadístico de Dickey-Fuller: {result[0]}")
print(f"Valor p: {result[1]}")
#Pasa la prueba en la primera diferenciacion, d = 1

#Encontrar el valor de la media movil(q)
fig, axes = plt.subplots(1, 2, sharex=False, figsize=(15, 6))
axes[0].plot(df_ajust['Close'].diff(), color='orangered'); axes[0].set_title('Primera diferenciacion')
axes[1].autoscale()  
axes[1].set_xscale('linear')
axes[1].set_yscale('linear')
plot_acf(df_ajust['Close'].diff().dropna(), ax=axes[1],color='orangered')
plt.title('Autocorrelacion simple')

# Ajustar el modelo ARIMA (diferenciación incluida en el modelo)
model = ARIMA(df_ajust['Close'], order=(7, 1, 20)) #(p,d,q)
#p es cuantos valores pasados se usan para predecir el actual (auto regresivo)
#d es cuantas veces la serie temporal ha sido estacionada (diferenciación)
#q es el número de terminos de media movil que se utilizan para modelar la 
#dependencia entre un error y los errores pasados
arima_result = model.fit()

# Resumen del modelo
print(arima_result.summary())

# Hacer predicciones (15 días futuros)
predicciones = arima_result.forecast(steps=15)

# Mostrar las predicciones
print(predicciones)

# Graficar los precios reales y las predicciones
plt.figure(figsize=(12,6))
plt.plot(df_ajust.index, df_ajust['Close'], label='Precios reales')
plt.plot(pd.date_range(df_ajust.index[-1], periods=15, freq='D'), predicciones,
         label='Predicciones', color='red')
plt.title('Predicción de precios de cierre')
plt.legend()
plt.show()
