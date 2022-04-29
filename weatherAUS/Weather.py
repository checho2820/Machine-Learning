# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 04:32:56 2022

@author: Sergio Gomez
"""
import math, time, random, datetime
import numpy as np
import pandas as pd
from warnings import simplefilter
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model

url = 'weatherAUS.csv'
data = pd.read_csv(url)

#Tratamiento de la data
    #Se eliminan las siguientes columnas, se consideran irrelevantes
data.drop(columns = ['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm'], axis = 1, inplace=True)    
data.drop(['Location','RISK_MM'],axis=1, inplace=True)

    #Los datos de fechas se han cambiado por el numero del mes en el que esta la tupla
data['Date'] = pd.to_datetime(data['Date']).dt.month

    #El nombre se ha cambiado de 'Date' a 'Month'
data.rename(columns={'Date':'Month'}, inplace=True)

    #Los valores de 'Yes' y 'No' será reemplazados por los valores numericos '1' y '2'
data.RainTomorrow.replace(['Yes', 'No'], [1, 0], inplace=True)
data.RainToday.replace(['Yes', 'No'], [1, 0], inplace=True)

    #Los nulos será eliminados al igual que los valores llamados 'Nan'
data.dropna(axis=0,how='any', inplace=True)

    #Los datos de las siguientes tablas seran reemplazados por valores numéricos
data.WindDir9am.replace(['W', 'NNW', 'SE', 'ENE', 'SW', 'SSE', 'S', 'NE', 
                         'SSW', 'N', 'WSW', 'ESE', 'E', 'NW', 'WNW', 'NNE'], 
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                        inplace=True)

data.WindDir3pm.replace(['WNW', 'WSW', 'E', 'NW', 'W', 'SSE', 'ESE', 'ENE', 
                         'NNW', 'SSW', 'SW', 'SE', 'N', 'S', 'NNE', 'NE'], 
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                        inplace=True)

data.WindGustDir.replace(['W', 'WNW', 'WSW', 'NE', 'NNW', 'N', 'NNE', 'SW', 
                         'ENE', 'SSE', 'S', 'NW', 'SE', 'ESE', 'E', 'SSW'], 
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                        inplace=True)

#Partimos la data en 2

data_train = data[:56463]
data_test = data[56463:]

x = np.array(data_train.drop(['RainTomorrow'], 1))
y = np.array(data_train.RainTomorrow)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)

x_test_out = np.array(data_test.drop(['RainTomorrow'], 1))
y_test_out = np.array(data_test.RainTomorrow)

#Regresión logística

#Seleccionar un modelo
logreg = LogisticRegression(solver='lbfgs', max_iter = 7600)

#Entreno el modelo
logreg.fit(x_train, y_train)

#Métricas
print('*'*50)
print('Regresión logística')

#Accuracy de entrenamiento de entrenamiento
print(f'accuracy de entrenamiento de entrenamiento: {logreg.score(x_train, y_train)}')

# Accuracy de test de entrenamiento
print(f'accuracy de test de entrenamiento: {logreg.score(x_test, y_test)}')

#Accuracy de validacion
print(f'acucuracy de validación: {logreg.score(x_test_out, y_test_out)}')




#Maquina de soporte vectorial


#Seleccionar un modelo
svc = SVC(kernel='rbf') #Se utiliza el núcleo rbf/gaussiano para adaptarse al modelo.

#Entreno el modelo
svc.fit(x_train, y_train)

#Métricas
print('*'*50)
print('Máquina de soporte vectorial')

#Accuracy de entrenamiento de entrenamiento
print(f'accuracy de entrenamiento de entrenamiento: {svc.score(x_train, y_train)}')

# Accuracy de test de entrenamiento
print(f'accuracy de test de entrenamiento: {svc.score(x_test, y_test)}')

#Accuracy de validacion
print(f'acucuracy de validación: {svc.score(x_test_out, y_test_out)}')



#Árbol de decisión 


#Seleccionar un modelo
arbol = DecisionTreeClassifier(max_depth=2, random_state=42)# Se usa un árbol de profundidad 2 para que no haya overfitting

#Entreno el modelo
arbol.fit(x_train, y_train)

#Métricas
print('*'*50)
print('Árbol de desición')

#Accuracy de entrenamiento de entrenamiento
print(f'accuracy de entrenamiento de entrenamiento: {arbol.score(x_train, y_train)}')

# Accuracy de test de entrenamiento
print(f'accuracy de test de entrenamiento: {arbol.score(x_test, y_test)}')

#Accuracy de validacion
print(f'acucuracy de validación: {arbol.score(x_test_out, y_test_out)}')



#Random Forest Classifier


#Seleccionar un modelo
clf = RandomForestClassifier(max_depth=2, random_state=0)

#Entreno el modelo
clf.fit(x_train, y_train)

#Métricas
print('*'*50)
print('Random Forest Classifier')

#Accuracy de entrenamiento de entrenamiento
print(f'accuracy de entrenamiento de entrenamiento: {clf.score(x_train, y_train)}')

# Accuracy de test de entrenamiento
print(f'accuracy de test de entrenamiento: {clf.score(x_test, y_test)}')

#Accuracy de validacion
print(f'acucuracy de validación: {clf.score(x_test_out, y_test_out)}')


#Regresión lineal


#Seleccionar un modelo
regr = linear_model.LinearRegression()

#Entreno el modelo
regr.fit(x_train, y_train)

#Métricas
print('*'*50)
print('Regresión lineal')

#Accuracy de entrenamiento de entrenamiento
print(f'accuracy de entrenamiento de entrenamiento: {regr.score(x_train, y_train)}')

# Accuracy de test de entrenamiento
print(f'accuracy de test de entrenamiento: {regr.score(x_test, y_test)}')

#Accuracy de validacion
print(f'acucuracy de validación: {regr.score(x_test_out, y_test_out)}')
