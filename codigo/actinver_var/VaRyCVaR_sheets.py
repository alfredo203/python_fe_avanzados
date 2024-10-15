
"""
Created on Wed Sep 18 17:55:05 2024

@author: 
    
Calculos de Valores VaR por el metodo historico y por simulacion Monte Carlo
"""

import pandas as pd #manejo de dataframes
import numpy as np #operaciones matematicas y generacion de num aleatorios
import matplotlib.pyplot as plt #creacion de graficos 
import yfinance as yf #obtencion de datos financieros 
import datetime as dt #trabajar con fechas


#getdata exatre los valores de cierre de las emisoras espificadas en tickers
#devuelve el cambio porcentual de la serie (rendimiento), la media del 
#cambio y la matriz de covarianza de las emisoras 
def getdata(stocks, start, end): 
    stockdata =yf.download(stocks, start=start, end=end) 
    stockdata = stockdata['Close'] 
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
initialPortfolio = 10000

def MonteCarlo(mc_sims, T, media_rendimiento, peso, initialPortfolio, covmatrix):
    # Verificar que media_rendimiento sea un vector
    meanM = np.tile(media_rendimiento, (T, 1))  # T filas y una copia de media_rendimiento en cada fila
    portfolio_sims = np.zeros((T, mc_sims))  # Inicializa la matriz para guardar simulaciones

    for m in range(mc_sims):
        # Generar rendimientos diarios simulados
        Z = np.random.normal(size=(T, len(peso)))
        L = np.linalg.cholesky(covmatrix)
        dailyReturns = meanM + Z @ L.T  # Multiplicación de matrices para simular rendimientos correlacionados

        # Cálculo de la evolución del portafolio
        portfolio_returns = np.cumprod(1 + np.dot(dailyReturns, peso))  # Acumular rendimientos diarios
        portfolio_sims[:, m] = portfolio_returns * initialPortfolio  # Aplicar valor inicial del portafolio
        
        plt.plot(portfolio_sims)
        plt.ylabel('Portfolio Value ($)')
        plt.xlabel('Days')
        plt.title('MC simulation of a stock portfolio')
        plt.show()

    return portfolio_sims
    

portfolio_sims = MonteCarlo(mc_sims, T, media_rendimiento, peso, initialPortfolio, covmatrix)

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
print("\nVaR:")

print(' historical VaR 95th CI   :    ', round(inversion_inicial*hVaR, 2))
print(" MC VaR  95th CI          :    ", round(VaR, 2))

print("\nCVaR:")

print(' historical CVaR 95th CI  :    ', round(inversion_inicial*hCVaR, 2))
print(" MC CVaR 95th CI          :    ", round(CVaR, 2))

print("\Portfolio")

print('initial portfolio         :    ', round(initialPortfolio))
print('portfolio performance     :    ', round(pRet*inversion_inicial))
