# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 16:05:51 2024

@author: edson

Obtener datos de Yahoo finance
"""

# Instalar yfinance, se pega en la terminal o se corre con ! al inicio
#pip install yfinance

# Importar librerias
import matplotlib.pyplot as plt
import yfinance as yf
import pygsheets 
import pandas as pd
import numpy as np

# Descargar información histórica
tickers = ["AA","AAL","AAPL","AMM.TO","ABBV","ABNB","ACTINVRB.MX","AC","AFRM",
           "AGNC","ALFAA.MX","ALPEKA.MX","ALSEA.MX","AMAT","AMD","AMX","AMZN",
           "APA","ASURB.MX","ATER","ATOS","AIY.DE","AVGO","AXP","BABA","BAC",
           "BA","BBAJIOO.MX","BIMBOA.MX","BMY","BNGO","CAT","CCL",
           "CEMEXCPO.MX","CHDRAUIB.MX","CLF","COST","CRM","CSCO",
           "CUERVO.MX","CVS","CVX","C","DAL","DIS","DVN","ELEKTRA.MX","ETSY",
           "FANG","FCX","FDX","FEMSAUBD.MX","FIBRAMQ12.MX","FIBRAPL14.MX",
           "FSLR","FUBO","FUBO","FUNO11.MX","F","GAPB.MX","GCARSOA1.MX","GCC",
           "GENTERA.MX","GE","GFINBURO.MX","GFNORTEO.MX","GILD","GMEXICOB.MX",
           "GME","GM","GOLD","GOOGL","GRUMAB.MX","HD","INTC","JNJ","JPM",
           "KIMBERA.MX","KOFUBL.MX","KO","LABB.MX",
          "LASITEB-1.MX","LCID","LIVEPOLC-1.MX","LLY","LUV","LVS","LYFT","MARA",
          "MARA","MA","MCD","MEGACPO.MX","MELIN.MX","META","MFRISCOA-1.MX","MGM",
          "MRK","MRNA","MRO","MSFT","MU","NCLHN.MX","NFLX","NKE","NKLA","NUN.MX",
          "NVAX","NVDA","OMAB.MX","ORBIA.MX","ORCL","OXY1.MX","PARA","PBRN.MX","PE&OLES.MX",
          "PEP","PFE","PG","PINFRA.MX","PINS","PLTR","PYPL","QCOM","Q.MX","RCL",
          "RIOT","RIVN","ROKU","RA.MX","SBUX","SHOP","SITES1A-1.MX","SKLZ",
          "SOFI","SPCE","SQ","TALN.MX","TERRA13.MX","TGT","TLEVISACPO.MX","TMO",
          "TSLA","TSMN.MX","TWLO","TX","T","UAL","UBER","UNH","UPST","VESTA.MX",
          "VOLARA.MX","VZ","V","WALMEX.MX","WFC","WMT","WYNN","XOM","X","ZM"] 

#Elegir fechas
# Pueden poner las fechas que gusten en el formato AAAA-MM-DD
inicio = "2022-03-01"
final = "2024-09-17"

#Crear el data frame y elegir variables
df = yf.download(tickers, start=inicio, end=final) #Esta es la base completa
datos_req = ["Close"] #Aqui pueden poner los datos que quieran usar
df_aj = df[datos_req] #Crea un df con los datos que quieren
df_aj.columns = df_aj.columns.droplevel(0) #elimina la fila Close



# Revisar si hay valores NaN 
print (f"cuantos nan hay en cada columna{df_aj.isna().sum()}")# Ver cuántos 
# valores NaN hay en cada columna


# Eliminar o rellenar los valores faltantes (NaN) 
df_act = df_aj.dropna()  # Eliminar filas con NaN este va a ser
# el que se convertirá en csv
print(f"Cuantos nan hay en cada columna de df_fin {df_act.isna().sum()}")
#cuantos nan hay en df_act


#Genera una gráfica para cada una de las acciones
#Como esta identado hará una gráfica para cada ticker
#for ticker in tickers: #Va a usar todos los tickers dentro de la lista tickers
    #plt.figure(figsize=(10,6)) #asi crea una grafica para cada accion
    #df_act[ticker].plot(label=ticker,color="purple")#Pueden cambiar c
    #plt.title(f"Precio de cierre {ticker}") #titulo 
    #plt.xlabel("Fecha") #eje x
    #plt.ylabel("Precio de cierre") #eje y
    #plt.legend() #muestra la leyenda
    #plt.grid(True) # hace una cuadricula
    #plt.show() #muestra el gráfico

#Pasar el dataframe a google sheets
# Autenticar usando el archivo de credenciales descargado de Google Cloud
#gc = pygsheets.authorize(service_file="C:/Users/edson/OneDrive/Documents/EqupoPython/credenciales.json")

#Abrir el google spreadsheet (dónde "BaseActinver es el nombre de mi hoja)
#sh = gc.open("BaseActinver")

#Seleccionar la primera hoja
#wks = sh[0]

#Actualizar la primera hoja con el df_act
#wks.set_dataframe(df_aj,(1,1),copy_index=True)

