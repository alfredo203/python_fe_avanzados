# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:33:45 2024

@author: David
"""

import numpy as np
import pandas as pd
import yfinance as yf
# Lista de tickers de acciones y ETFs que se analizarán
tickers = ["AA", "AAL", "AAPL", "AMM.TO", "ABBV", "ABNB", "ACTINVRB.MX", "AC", "AFRM",
           "AGNC", "ALFAA.MX", "ALPEKA.MX", "ALSEA.MX", "AMAT", "AMD", "AMX", "AMZN",
           "APA", "ASURB.MX", "ATER", "ATOS", "AIY.DE", "AVGO", "AXP", "BABA", "BAC",
           "BA", "BBAJIOO.MX", "BIMBOA.MX", "BMY", "BNGO", "CAT", "CCL",
           "CEMEXCPO.MX", "CHDRAUIB.MX", "CLF", "COST", "CRM", "CSCO",
           "CUERVO.MX", "CVS", "CVX", "C", "DAL", "DIS", "DVN", "ELEKTRA.MX", "ETSY",
           "FANG", "FCX", "FDX", "FEMSAUBD.MX", "FIBRAMQ12.MX", "FIBRAPL14.MX",
           "FSLR", "FUBO", "FUNO11.MX", "F", "GAPB.MX", "GCARSOA1.MX", "GCC",
           "GENTERA.MX", "GE", "GFINBURO.MX", "GFNORTEO.MX", "GILD", "GMEXICOB.MX",
           "GME", "GM", "GOLD", "GOOGL", "GRUMAB.MX", "HD", "INTC", "JNJ", "JPM",
           "KIMBERA.MX", "KOFUBL.MX", "KO", "LABB.MX", "LASITEB-1.MX", "LCID",
           "LIVEPOLC-1.MX", "LLY", "LUV", "LVS", "LYFT", "MARA", "MA", "MCD", 
           "MEGACPO.MX", "MELIN.MX", "META", "MFRISCOA-1.MX", "MGM", "MRK", 
           "MRNA", "MRO", "MSFT", "MU", "NCLHN.MX", "NFLX", "NKE", "NKLA", 
           "NUN.MX", "NVAX", "NVDA", "OMAB.MX", "ORBIA.MX", "ORCL", "OXY1.MX", 
           "PARA", "PBRN.MX", "PE&OLES.MX", "PEP", "PFE", "PG", "PINFRA.MX", 
           "PINS", "PLTR", "PYPL", "QCOM", "Q.MX", "RCL", "RIOT", "RIVN", 
           "ROKU", "RA.MX", "SBUX", "SHOP", "SITES1A-1.MX", "SKLZ", "SOFI", 
           "SPCE", "SQ", "TALN.MX", "TERRA13.MX", "TGT", "TLEVISACPO.MX", 
           "TMO", "TSLA", "TSMN.MX", "TWLO", "TX", "T", "UAL", "UBER", "UNH", 
           "UPST", "VESTA.MX", "VOLARA.MX", "VZ", "V", "WALMEX.MX", "WFC", 
           "WMT", "WYNN", "XOM", "X", "ZM"]

# Fechas de inicio y final para los datos a descargar
inicio = "2022-03-01"
final = "2024-09-17"

# Función para descargar los datos de cierre ajustado de los tickers seleccionados
def descargar_datos(tickers, inicio, final, datos_req=["Close"]):
    # Descarga los datos de Yahoo Finance en el rango de fechas especificado
    df = yf.download(tickers, start=inicio, end=final)
    # Devuelve solo las columnas solicitadas (por defecto, el precio de cierre ajustado)
    return df[datos_req]

# Función para calcular los rendimientos logarítmicos
def calcular_rendimientos_log(df_aj):
    # Calcula los rendimientos simples porcentuales
    rendimiento = df_aj.pct_change()
    # Convierte los rendimientos simples en rendimientos logarítmicos
    log_returns = np.log(1 + rendimiento)
    # Retorna los rendimientos logarítmicos
    return log_returns

# Función para simular múltiples portafolios
def simular_portafolios(log_returns, num_portafolios=5000):
    # Obtiene el número de activos en el portafolio
    num_activos = log_returns.shape[1]
    # Inicializa matrices para almacenar los pesos, rendimientos, volatilidades y ratios Sharpe
    weight = np.zeros((num_portafolios, num_activos))
    ReturnEsp = np.zeros(num_portafolios)
    VolEsp = np.zeros(num_portafolios)
    RadioSharpe = np.zeros(num_portafolios)

    # Calcula la media y covarianza de los rendimientos logarítmicos
    meanlogReturns = log_returns.mean()
    Sigma = log_returns.cov()

    # Simulación de los portafolios
    for k in range(num_portafolios):
        # Genera pesos aleatorios para los activos y los normaliza a 1
        w = np.random.random(num_activos)
        w /= np.sum(w)
        # Almacena los pesos en la matriz correspondiente
        weight[k, :] = w

        # Calcula el rendimiento esperado del portafolio con los pesos aleatorios
        ReturnEsp[k] = np.sum(meanlogReturns * w)
        # Calcula la volatilidad esperada del portafolio
        VolEsp[k] = np.sqrt(np.dot(w.T, np.dot(Sigma, w)))
        # Calcula el Ratio Sharpe del portafolio
        RadioSharpe[k] = ReturnEsp[k] / VolEsp[k]

    # Retorna las matrices de pesos, rendimientos, volatilidades y ratios Sharpe
    return weight, ReturnEsp, VolEsp, RadioSharpe

# Función para identificar el mejor portafolio según el Ratio Sharpe
def encontrar_mejor_portafolio(weight, ReturnEsp, VolEsp, RadioSharpe):
    # Encuentra el índice del portafolio con el mayor Ratio Sharpe
    max_index = RadioSharpe.argmax()
    # Obtiene los pesos, rendimiento, volatilidad y Ratio Sharpe del mejor portafolio
    best_weights = weight[max_index, :]
    return best_weights, ReturnEsp[max_index], VolEsp[max_index], RadioSharpe[max_index]

# Función para mostrar los resultados del mejor portafolio
def mostrar_resultados(tickers, best_weights, retorno, volatilidad, sharpe_ratio):
    # Muestra los pesos de los activos en el mejor portafolio
    print("Mejores pesos del portafolio:")
    for i, ticker in enumerate(tickers):
        print(f"{ticker}: {best_weights[i] * 100:.2f}%")

    # Muestra el rendimiento esperado, la volatilidad y el Ratio Sharpe del mejor portafolio
    print(f"Retorno esperado del portafolio: {retorno:.4f}")
    print(f"Volatilidad esperada del portafolio: {volatilidad:.4f}")
    print(f"Ratio Sharpe máximo: {sharpe_ratio:.4f}")

# Función principal que ejecuta todo el análisis de portafolios
def ejecutar_analisis(tickers, inicio, final):
    # Descarga los datos de los tickers seleccionados en el rango de fechas indicado
    df_aj = descargar_datos(tickers, inicio, final)
    # Calcula los rendimientos logarítmicos a partir de los precios de cierre ajustados
    log_returns = calcular_rendimientos_log(df_aj)
    # Simula portafolios aleatorios basados en los rendimientos logarítmicos
    weight, ReturnEsp, VolEsp, RadioSharpe = simular_portafolios(log_returns)
    # Encuentra el portafolio con el mayor Ratio Sharpe
    best_weights, retorno, volatilidad, sharpe_ratio = encontrar_mejor_portafolio(weight, ReturnEsp, VolEsp, RadioSharpe)
    # Muestra los resultados del mejor portafolio
    mostrar_resultados(tickers, best_weights, retorno, volatilidad, sharpe_ratio)

# Ejecuta el análisis de portafolios con los tickers y fechas indicados
ejecutar_analisis(tickers, inicio, final)
