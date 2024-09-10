# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 17:04:15 2024

@author: david

Primer modelo nivel-log competencia
"""
#Importar librerias
import pandas as pd #Para manejar tablas
import seaborn as sns #Hacer gráficas 
import matplotlib.pyplot as plt #contruccion de graficas 
import statsmodels.api as sm #regresion y resumen estadsitico 

#Close: cierre, valor de la accion 
#precio ajustado: cambios del cierre a la mañana

#Definimos path como la ruta donde esta nuestra base
path = 'C:/Users/david/Documents/Clase_Prob/Data/yahoo_stock_train.csv'
df = pd.read_csv(path)


# Definimos las variables
x = df[["High","Low"]] #variables independientes o explicativas
#Escogimos unicamente estas dos dado su poder explicativo 
y = df["Close"] #Variable dependiente

#Definimos las variables de forma independiente 
#para poder graficar cada una de estas 
a = df['Close']
b = df['High']
c = df['Low']

#Grafica de lineas de todas las variables juntas
#para determinar la relacion entre las tendencias
plt.subplots()

plt.plot(a) #grafica de lineas precio de cierre
plt.plot(b) #grafica precio maximo 
plt.plot(c) #grafica precio minimo 

plt.show()

#Conjunto de graficas por variable 
plt.figure()
plt.subplot(221)
plt.plot(a)
plt.title('Precio de cierre')

plt.subplot(222)
plt.plot(b)
plt.title('Varlos máximo')

plt.subplot(223)
plt.plot(c)
plt.title('Valor mínimo')

plt.show()

# Pairplot de Seaborn con rectas de regresión
sns.pairplot(df, x_vars=["High","Low"],
             y_vars="Close", height=5, aspect=0.7,
             kind='reg')#Hace gráficas con regresión 

#Generemos el modelo y resumen estadsitico 
model = sm.OLS(y, x).fit()
print(model.summary())

y_predict = model.predict(x)