# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 09:46:49 2024

@author: Admin
"""
#importar librerias 
import numpy as np
np.random.seed(4)
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

#importar los datos 
datos = pd.read_csv(r"C:\Users\Admin\Documents\yahoo_stock_train.csv", index_col='Date', parse_dates=['Date'])
datos = datos.dropna()
datos = datos.sort_index()


#crear sets de entrenamiento utilizando iloc de pandas
#.iloc sirve para hacer un indexado basado en posiciones enteras usando numeros que representan los indices que en este caso es la columna High
 
set_entrenamiento = datos[:'2019-01-01 00:00:00'].iloc[:,1:2]
set_validacion = datos['2020-01-01 00:00:00':].iloc[:,1:2]


# utilizamos sklern para normalizar valores 
sc = MinMaxScaler(feature_range=(0,1))
set_entrenamiento_escalado = sc.fit_transform(set_entrenamiento)

#tomamos bloques de 10 datos 
time_step = 10
x_tren = []
y_tren = []
m = len(set_entrenamiento_escalado)

#usamos for para dividir en bloques de 10 y almacenarlos en xtren y ytren
for i in range(time_step,m):
    x_tren.append(set_entrenamiento_escalado[i-time_step:i,0])
    y_tren.append(set_entrenamiento_escalado[i-time_step:i,0])

x_tren, y_tren = np.array(x_tren), np.array(y_tren)

#ajustamos los sets para la entrada del modelo con reshape
x_tren = np.reshape(x_tren, (x_tren.shape[0], x_tren.shape[1],1))

#establecemos la entrada y salida de datos 
dim_entrada = (x_tren.shape[1],1)
dim_salida = 1
na = 100 #numero de neuronas 

#el tipo de modelo es secuencial 
modelo = Sequential()
modelo.add(LSTM(units=na, input_shape=dim_entrada))

#establecemos los parametros 
modelo.add(Dense(units=dim_salida))

modelo.compile(optimizer='rmsprop', loss='mse')

modelo.fit(x_tren,y_tren,epochs=40,batch_size=32)
#epochs son la iteraciones y batch es la muestra de datos 

#colocamos los test para la validacion de valores 
x_test = set_validacion.values
x_test = sc.transform(x_test)

X_test = []
for i in range(time_step,len(x_test)):
    X_test.append(x_test[i-time_step:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

#establecemos la prediccion 

prediccion = modelo.predict(X_test)
prediccion = sc.inverse_transform(prediccion)

# Convertir set_validacion a un array numpy para hacer la comparación
valores_reales = set_validacion.values[time_step:]

# Graficar los valores reales y las predicciones
plt.figure(figsize=(10,6))

# Graficar los valores reales
plt.plot(valores_reales, color='blue', label='Valor Real')

# Graficar las predicciones
plt.plot(prediccion, color='red', label='Predicción')

# Añadir título y leyenda
plt.title('Predicción de Precios de Acciones')
plt.xlabel('Días')
plt.ylabel('Precio de la Acción')
plt.legend()

# Mostrar la gráfica
plt.show()


# Últimos datos disponibles del conjunto de validación que usarás para predecir
ultimo_bloque = set_validacion.values[-time_step:]  
ultimo_bloque_escalado = sc.transform(ultimo_bloque)  

# Inicializar una lista para almacenar las predicciones
predicciones_futuras = []

# Hacer predicciones para los próximos 15 días
for i in range(15):
    # Preparar los datos de entrada para el modelo (debe ser de la forma (1, time_step, 1))
    x_input = np.reshape(ultimo_bloque_escalado, (1, time_step, 1))

    # Hacer la predicción
    prediccion = modelo.predict(x_input)

    # Invertir la normalización de la predicción
    prediccion_invertida = sc.inverse_transform(prediccion)

    # Guardar la predicción
    predicciones_futuras.append(prediccion_invertida[0, 0])

    # Añadir la predicción escalada al bloque para la próxima predicción
    nueva_prediccion_escalada = sc.transform(prediccion_invertida)
    
    # Actualizar el último bloque para incluir la nueva predicción y eliminar el primer dato
    ultimo_bloque_escalado = np.append(ultimo_bloque_escalado[1:], nueva_prediccion_escalada).reshape(time_step, 1)

# Convertir la lista de predicciones a un array de numpy
predicciones_futuras = np.array(predicciones_futuras)

# Mostrar las predicciones de los próximos 15 días
print("Predicciones para los próximos 15 días:", predicciones_futuras)

# Graficar las predicciones
plt.figure(figsize=(10, 6))
plt.plot(predicciones_futuras, color='green', label='Predicciones Futuras (15 días)')
plt.title('Predicción de Precios de Acciones para los Próximos 15 Días')
plt.xlabel('Días')
plt.ylabel('Precio de la Acción')
plt.legend()
plt.show()
