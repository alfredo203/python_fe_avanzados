# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 16:05:51 2024

@author: edson

Obtener datos de Yahoo finance
"""

# Instalar yfinance, se pega en la terminal o se corre con ! al inicio
#!pip install yfinance

# Importar librerias
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Descargar información histórica
tickers = ["GOOG","BKNG","META", "AAPL","TSLA","^IRX"] #Son los simbolos con
#los que se identifican las acciones, se pueden agregar más desde aquí
tickers_sin_bkng = tickers.copy() #Copia la lista para remover BKNG
tickers_sin_bkng.remove("BKNG") #Solo es para ver la Grafica mejor

#Elegir fechas
# Pueden poner las fechas que gusten en el formato AAAA-MM-DD
inicio = "2017-01-01"
final = "2024-09-12"

#Crear el data frame y elegir variables
df = yf.download(tickers, start=inicio, end=final) #Esta es la base completa
datos_req = ["Close"] #Aqui pueden poner los datos que quieran usar
df_aj = df[datos_req] #Crea un df con los datos que quieren, este va a ser
# el que se convertirá en csv

# Generar gráfica de todos los tickers
for ticker in tickers: #Va a usar todos los tickers dentro de la lista tickers
    df_aj["Close", ticker].plot(label=ticker) # Crea la gráfica de cierre
plt.title("Precios de Cierre de las Acciones") #título
plt.xlabel("Fecha") #eje x
plt.ylabel("Precio de Cierre") # eje y
plt.legend() # Mostrar la leyenda
plt.grid(True) # Hace una cuadricula
plt.show() # Muestra 

# Genera unag gráfica de los datos sin BKNG para verlos mejor
for ticker in tickers_sin_bkng: #usa todos los tickers dentro de la lista
    df_aj["Close", ticker].plot(label=ticker) #Crea la gráfica de cierre
plt.title("Precios de Cierre de las Acciones") #titulo
plt.xlabel("Fecha") # eje x
plt.ylabel("Precio de Cierre") # eje y
plt.legend()  # Mostrar leyenda
plt.grid(True) #hace una cuadrícula
plt.show() #muestra el gráfico

#Genera una gráfica para cada una de las acciones
#Como esta identado hará una gráfica para cada ticker
for ticker in tickers: #Va a usar todos los tickers dentro de la lista tickers
    plt.figure(figsize=(10,6)) #asi crea una grafica para cada accion
    df_aj["Close",ticker].plot(label=ticker,color="purple")#Pueden cambiar c
    plt.title(f"Precio de cierre {ticker}") #titulo 
    plt.xlabel("Fecha") #eje x
    plt.ylabel("Precio de cierre") #eje y
    plt.legend() #muestra la leyenda
    plt.grid(True) # hace una cuadricula
    plt.show() #muestra el gráfico


# Convertir el df a un archivo de Excel
# Usa pwd para ver en que carpeta se guardará el archivo

#df_aj.to_csv("Export/Base Actinver.csv") 
#Export es el nombre de la subcarpeta donde queremos guardar el archivo
#Base Actinver el nombre del archivo

#Falta saber como subirlo a Google sheets :(




