#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 00:04:11 2024

@author: alfredo
"""

## Regresión ejemplo

from sklearn import linear_model

reg = linear_model.LinearRegression()

x = [[1,2,3,4,5]]

y = [1,2,3,4,5]


reg.fit(x,y)
reg.coef_


## Regresión completa

import statsmodels.api as sm
import pandas as pd

df = pd.read_csv('data/Housing.csv', sep=',')

#define response variable
y = df['price']

#define predictor variables
x = df[['area']]

#add constant to predictor variables
x = sm.add_constant(x)

#fit linear regression model
model = sm.OLS(y, x).fit()

#view model summary
print(model.summary())
