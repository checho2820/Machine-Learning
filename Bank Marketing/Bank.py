# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 03:22:29 2022

@author: Sergio Gomez
"""
import numpy as np
import pandas as pd
from warnings import simplefilter
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model

url = 'bank-full.csv'
data = pd.read_csv(url)

#Tratamiento de la data 
    #Los valores de SI y NO, serán cambiados a numeros
data.default.replace(['yes', 'no'], [1, 0], inplace=True)
data.housing.replace(['yes', 'no'], [1, 0], inplace=True)
data.loan.replace(['yes', 'no'], [1, 0], inplace=True)
data.y.replace(['yes', 'no'], [1, 0], inplace=True)
    
    #Se eliminan las siguientes columnas, se consideran irrelevantes
data.drop(columns = ['job', 'marital', 'education','contact', 'month', 'poutcome'], axis=1, inplace=True)

data_train = data[:26606]
data_test = data[26606:]

    #Normalizamos la data
x = np.array(data_train.drop(['y'], 1))
y = np.array(data_train.y)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)

x_test_out = np.array(data_test.drop(['y'], 1))
y_test_out = np.array(data_test.y)


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