# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 18:46:56 2022

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

url = 'diabetes.csv'
data = pd.read_csv(url)

#Tratamiento de la data
    #Sabemos que hay 768 personas con una distribucion desigual
    #Tambien hay algunos valores de (0) en las tablas de 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin' y 'BMI'.
    #Los datos que estan definidos como '0' serán reemplazados por el valor promedio del grupo de datos.
  

#Los datos '0' de las tablas mencionadas anteriormente han pasado a ser el valor del promedio de cada respectivo dato
    
    #El promedio de la tabla Glucose es 121, entonces los valores de 0 seran reemplazdos por 121
data.Glucose.replace(0, 121, inplace=True)
    
    #El promedio de la tabla BloodPressure es 69, entonces los valores de 0 seran reemplazdos por 69
data.BloodPressure.replace(0, 69, inplace=True)

    #El promedio de la tabla SkinThickness es 21, entonces los valores de 0 seran reemplazdos por 21
data.SkinThickness.replace(0, 21, inplace=True) 

    #El promedio de la tabla Insulin es 80, entonces los valores de 0 seran reemplazdos por 80
data.Insulin.replace(0, 80, inplace=True) 
 
    #El promedio de la tabla BMI es 32, entonces los valores de 0 seran reemplazdos por 32
data.BMI.replace(0, 32, inplace=True) 

    
#Partir la data en dos

data_train = data[:385]
data_test = data[385:]

x = np.array(data_train.drop(['Outcome'], 1))
y = np.array(data_train.Outcome) #0 no tiene diabetes, 1 si tiene diabetes

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

x_test_out = np.array(data_test.drop(['Outcome'], 1))
y_test_out = np.array(data_test.Outcome)
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


 
