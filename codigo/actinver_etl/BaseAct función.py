# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 13:55:55 2024

@author: edson
"""

#La función actbase arroja un dataframe con información histórica de los 
#precios de cierre de acciones que quieras, extraidas de yahoo finance
#la función requiere tickers, inicio, final
#tickers es como se encuentran las empresas en yahoo finance ej:"NVDA","META"
#inicio es la fecha desde la que quieres que comience tu dataframe y debe estar
#en formato %Y-%m-%d
#final es la fecha hasta la que quieres que termine tu dataframe y debe estar
#en formato %Y-%m-%d
def actbase (tickers, inicio, final):
    import yfinance as yf
    import pandas as pd
    df = yf.download(tickers, start=inicio, end=final) #Esta es la base completa
    datos_req = ["Close"] #Aqui pueden poner los datos que quieran usar
    df_aj = df[datos_req] #Crea un df con los datos que quieren
    if isinstance(df_aj.columns, pd.MultiIndex):
        df_aj.columns = df_aj.columns.droplevel(0)  # Elimina el primer nivel si es multiíndice
    
    return df_aj
#Ejemplo:
# tickers =["META","NVDA"] crear una lista llamada tickers, y poner el ticker
# de la empresa que quieras buscar
# inicio= "2017-01-01" #crear la variable inicio como string en el formato:
    #%Y-%m-%d
# final= "2024-10-07" #crear la variable inicio como string en el formato:
    #%Y-%m-%d
# dataframe = actbase(tickers, inicio, final) #Creamos una variable que llame
# a la función, y nos generará un dataframe en la variable
#otro ejemplo
#data = actbase("NVDA", "2023-01-01", "2024-10-14")

