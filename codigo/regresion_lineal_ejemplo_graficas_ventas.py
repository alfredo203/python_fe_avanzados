#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 00:04:11 2024
Modificado Sep 05 2024

@author: RDavid
"""

## Regresión completa
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd

df = pd.read_csv('C:/Users/David/Downloads/advertising_training.csv', sep=',')

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
a = df['Sales']
b = df['Radio']
c = df['TV']
d = df['Newspaper']

sns.regplot(x=b, y=a, data=df, ci=None,
            line_kws={'color': 'red'}, scatter_kws={'color': 'red'})

sns.regplot(x=c, y=a, data=df, ci=None,
            line_kws={'color': 'blue'}, scatter_kws={'color': 'blue'})

sns.regplot(x=d, y=a, data=df, ci=None,
            line_kws={'color': 'green'}, scatter_kws={'color': 'green'})

# nombres del gráfico
plt.xlabel('Gastos en Medios')
plt.ylabel('Ventas')
plt.title('Relación entre Gastos y Ventas')
plt.show()