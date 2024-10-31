# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 14:49:17 2024

@author: alfre
"""

import pandas as pd
from sklearn.linear_model import LinearRegression as lm
import matplotlib.pyplot as plt

## LM con variables categoricas

path = r"C:\Users\alfre\Documents\GitHub\python_fe_basicos\data\diamonds.csv"
df = pd.read_csv(path)

## Stats describe
df.describe()

## Elegir una variable categorica y una numerica
model = lm()

x = df[['carat''color', 'cut']] ## construyendo df
y = df['price']  ## construyendo serie o vector

## asignamos la funcion para hacer la regresion al objeto model
model = lm()

# Cambiar tipo de variable. Método dummy (var categorica)
X = pd.get_dummies(data=x, drop_first=True,    dtype= 'int')

## Calcula con el vector "y" y el dataframe x la correlacion R
model.fit(X,y) ## fit significa ajuste (calculo)

## Muestra cual es la correlacion entre x y y
model.score(X, y)
print(model.intercept_, model.coef_, model.score(X, y))

## Ajustamos para crear un vector "linea"
y_fit = model.predict(X)

## Ahora todo junto (ejecutar en conjunto o por bloque)
# %%
plt.scatter(X['carat'], y)

# Agregar etiquetas
plt.xlabel('Carat')
plt.ylabel('Precio')
plt.title('Gráfica de dispersión precio vs carats')

plt.plot(X['carat'], y_fit, color='red')
plt.show()


## SM example

# Load modules and data
import statsmodels.api as sm

x = df.loc[:, df.columns != 'price'] ## construyendo df
y = df['price']  ## construyendo serie o vector

# Cambiar tipo de variable. Método dummy (var categorica)
X = pd.get_dummies(data=x, 
                   #columns= ['cut', 'color'], 
                   drop_first=True, 
                   dtype=int)

# Fit and summarize OLS model
mod = sm.OLS(y, X).fit()
mod.summary()

fig = sm.graphics.plot_partregress_grid(mod)
fig.tight_layout(pad=1.0)
