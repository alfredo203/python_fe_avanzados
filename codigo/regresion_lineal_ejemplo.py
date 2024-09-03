#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 00:04:11 2024

@author: alfredo
"""

## Regresión ejemplo

from sklearn import linear_model # Esto carga la libreria del MCO
import matplotlib.pyplot as plt 

reg = linear_model.LinearRegression() # Asigno una funcion a reg

x = [[1,2,3,4,5]]  # construimos X un data frame

y = [1,2,3,4,5] # construimos y que es un vector

# Ajusta la regresion
reg.fit(x,y)
reg.coef_ # Regresa el coeficiente.


## Regresión completa

import statsmodels.api as sm
import pandas as pd

df = pd.read_csv('data/advertising_training.csv', sep=',')

#define response variable
y = df['Sales']

#define predictor variables
x = df[['TV','Radio', 'Newspaper']]

#fit linear regression model
model = sm.OLS(y, x).fit()

#view model summary
print(model.summary())

## Ajustamos para crear un vector "linea"
y_fit = model.predict(x)


## Graficando mi mejor variable predictiva
import numpy as np
y = df['Sales']
x = df['Radio']

# fitting a linear regression line
m, b = np.polyfit(x, y, 1) 

plt.scatter(x,y)
# adding the regression line to the scatter plot
plt.plot(x, m*x + b)


