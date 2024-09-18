# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 16:05:51 2024

@author: edson

Obtener datos de Yahoo finance
"""

# Instalar yfinance, se pega en la terminal o se corre con ! al inicio
#!pip install yfinance

# Importar librerias
import matplotlib.pyplot as plt
import yfinance as yf
import pygsheets 

# Descargar información histórica
tickers = ["GOOG","BKNG","META", "AAPL","TSLA","^IRX"] #Son los simbolos con
#los que se identifican las acciones, se pueden agregar más desde aquí
tickers_sin_bkng = tickers.copy() #Copia la lista para remover BKNG
tickers_sin_bkng.remove("BKNG") #Solo es para ver la Grafica mejor

#Elegir fechas
# Pueden poner las fechas que gusten en el formato AAAA-MM-DD
inicio = "2018-01-01"
final = "2024-09-17"

#Crear el data frame y elegir variables
df = yf.download(tickers, start=inicio, end=final) #Esta es la base completa
datos_req = ["Close"] #Aqui pueden poner los datos que quieran usar
df_aj = df[datos_req] #Crea un df con los datos que quieren


# Revisar si hay valores NaN 
print (f"cuantos nan hay en cada columna{df_aj.isna().sum()}")# Ver cuántos 
# valores NaN hay en cada columna


# Eliminar o rellenar los valores faltantes (NaN) 
df_act = df_aj.dropna()  # Eliminar filas con NaN este va a ser
# el que se convertirá en csv
print(f"Cuantos nan hay en cada columna de df_fin {df_act.isna().sum()}")
#cuantos nan hay en df_act


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

#Pasar el dataframe a google sheets
# Autenticar usando el archivo de credenciales descargado de Google Cloud
gc = pygsheets.authorize(service_file="C:/Users/edson/OneDrive/Documents/EqupoPython/credenciales.json")

#Abrir el google spreadsheet (dónde "BaseActinver es el nombre de mi hoja)
sh = gc.open("BaseActinver")

#Seleccionar la primera hoja
wks = sh[0]

#Actualizar la primera hoja con el df_act
wks.set_dataframe(df_act,(1,1),copy_index=True)



