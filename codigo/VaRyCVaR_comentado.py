# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 17:55:05 2024

@author: 
    
Calculos de Valores VaR por el metodo historico y por simulacion Monte Carlo
"""

import pandas as pd #manejo de dataframes
import numpy as np #operaciones matematicas y generacion de num aleatorios
import datetime as dt #trabajar con fechas
import yfinance as yf #obtencion de datos financieros 
import matplotlib.pyplot as plt #creacion de graficos 

#getdata exatre los valores de cierre de las emisoras espificadas en tickers
#devuelve el cambio porcentual de la serie (rendimiento), la media del 
#cambio y la matriz de covarianza de las emisoras 
def getdata(stocks, start, end): 
    stockdata =yf.download(stocks, start=start, end=end) 
    stockdata = stockdata['Close'] #falta implementar la compatibilidad con 
    #google sheets
    rendimiento = stockdata.pct_change() 
    media_rendimiento = rendimiento.mean()
    covmatrix = rendimiento.cov()
    return rendimiento, media_rendimiento, covmatrix 

#desempeno calcula el rendimiento de nuestro portafolio y la desviacion 
#estandar del mismo 
def desempeno(peso, media_rendimiento, covmatrix, time):
    rendimiento = np.sum(media_rendimiento*peso)*time 
    std = np.sqrt(np.dot(peso.T, np.dot(covmatrix, peso))) * np.sqrt(time)
    return rendimiento, std 

#Establecemos los parametros para operar nuestras funciones
tickers = ["GOOG","BKNG","META", "AAPL","TSLA","^IRX"] #creamos un portafolio

#Establecemos el periodo de tiempo con el que trabajaremos
end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days=800)

#Utiizamos la funcion getdata para obtener nuestras tres primeras variables
rendimiento, media_rendimiento, covmatrix = getdata(tickers, start=start_date, 
                                                    end=end_date)
rendimiento = rendimiento.dropna() #eliminamos los valores nulos 

#Establecemos el peso correspondiente a cada accion en el portafolio 
peso = np.array([0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666])
peso /= np.sum(peso) #redondeamos los pesos para que sumen 1

#creamos una columna para almacenar el cambio diario de nuestro portafolio 
rendimiento['portafolio'] = rendimiento.dot(peso) #dot calcula el producto 
#de los vectores

#Metodo Historico para el calculo del VaR

#historicalVar toma los datos en el df rendimiento y con relacion un valor
#alpha (intervalo de confianza del 95%) regresa el percentil en el que se 
#encuentra el valor en riesgo maximo dado el periodo de tiempo establecido
def historicalVar(rendimiento, alpha=5):
    if isinstance(rendimiento, pd.Series):
        return np.percentile(rendimiento,alpha)
    elif isinstance(rendimiento, pd.DataFrame):
        return rendimiento.aggregate(historicalVar, alpha=alpha)
    else :
        raise TypeError('Se espera que rendimiento sea dataframe o serie')
        
#calcula el valor VaR historico condicional (los valores que exceden la medida
#del VaR historico) dado un intervalo de confianza alpha
def historicalCVar(rendimiento, alpha=5):
    if isinstance(rendimiento, pd.Series):
        belowVar = rendimiento <= historicalVar(rendimiento, alpha=alpha)
        return rendimiento[belowVar].mean()
    elif isinstance(rendimiento, pd.DataFrame):
        return rendimiento.aggregate(historicalCVar, alpha=alpha)
    else:
        raise TypeError('Se espera que rendimiento sea dataframe o serie')
   
#Establecemos el periodo de tiempo para el calculor del VaR
time = 100 #periodo de tiempo en dias 

#Aplicamos las funciones antes definidas para calcular hVaR y hCvaR
hVaR = -historicalVar(rendimiento['portafolio'], alpha=5)*np.sqrt(time)
hCVaR = -historicalCVar(rendimiento['portafolio'], alpha=5)*np.sqrt(time)

#Adicionalmente calculamos el rendimiento de nuestro portafolio y su
#desviacion estandar con la funcion desempeno 
pRet, pStd = desempeno(peso, media_rendimiento, covmatrix, time)

#Establecemos el valor monetario de nuestro portafolio 
inversion_inicial = 10000

#Imprimimos los resultados del Rendimiento total y los calculos de los VaR 
#historicos 
print('Rendimiento esparado del portafolio:      ', 
      round(inversion_inicial*pRet,2))
print('Value at Risk 95th CI    :      ', round(inversion_inicial*hVaR,2))
print('Conditional VaR 95th CI  :      ', round(inversion_inicial*hCVaR,2))        
        

# VaR por el metodo Monte Carlo 
mc_sims = 400 # numero de simulaciones
T = 100 #p eriodo de tiempo en dias 


meanM = np.full(shape=(T, len(peso)), fill_value=media_rendimiento)
meanM = meanM.T

portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)

initialPortfolio = 10000

for m in range(0, mc_sims):
    # MC loops
    #Genera una matriz de numeros aleatorios siguiendo una distribucion normal 
    #para simular choques aletorios en los rendimientos de los activos en cada 
    #periodo
    Z = np.random.normal(size=(T, len(peso)))
    #utiliza la descomposicion de cholesky sobre la matriz de covarianza 
    #modela las correlaciones entre los activos, transforma las variables
    #aleatorias no correlacionadas en variables correlacionadas 
    L = np.linalg.cholesky(covmatrix)
    #simulacion de rendimientos diarios, calcula la suma de los rendmientos 
    #promedio mas el choque aleatorio correlacionado
    dailyReturns = meanM + np.inner(L, Z)
    #rendimiento acumulado del portafolio, aplica los pesos del portafolio a 
    #los rendimientos diarios simulados y luego acumula los rendimientos 
    #con np.cumprod
    portfolio_sims[:,m] = np.cumprod(np.inner(peso, 
                                    dailyReturns.T)+1)*initialPortfolio
     
#graficamos las simulaciones 
plt.plot(portfolio_sims)
plt.ylabel('Portfolio Value ($)')
plt.xlabel('Days')
plt.title('MC simulation of a stock portfolio')
plt.show()

#Mcvar calcula el valor en riesgo historico para los datos optenidos en la
#simulacion montecarlo 
def mcVaR(rendimiento, alpha=5):
    if isinstance(rendimiento, pd.Series):
        return np.percentile(rendimiento, alpha)
    else:
        raise TypeError("Se espera una serie de datos de pandas")

def mcCVaR(rendimiento, alpha=5):
    if isinstance(rendimiento, pd.Series):
        belowVaR = rendimiento <= mcVaR(rendimiento, alpha=alpha)
        return rendimiento[belowVaR].mean()
    else:
        raise TypeError("Se espera una serie de datos de pandas")
     
#se selecciona la ultima fila de portfolio_sims que contiene los valores 
#finales, despues de todas las simulaciones
portResults = pd.Series(portfolio_sims[-1,:])

#utilizamos las funciones mcvar y mccvar para calcular los valores var
VaR = initialPortfolio - mcVaR(portResults, alpha=5)
CVaR = initialPortfolio - mcCVaR(portResults, alpha=5)

#imprimimos y presentanmos los resultados 
print('VaR ${}'.format(round(VaR,2)))
print('CVaR ${}'.format(round(CVaR,2)))
