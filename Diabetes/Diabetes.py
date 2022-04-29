# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 18:46:56 2022

@author: Sergio Gomez
"""
import numpy as np
import pandas as pd
from warnings import simplefilter
from sklearn.model_selection import train_test_split 

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

x = np.array()
