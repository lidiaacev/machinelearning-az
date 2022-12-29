#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 19:43:11 2019

@author: juangabriel
"""

# Plantilla de Pre Procesado

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values


# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

##se escalan (normaliza o estandariza) todas las variables predictoras para que todas jueguen con el mismo peso. En el caso de la variable que quermeos predecir,
##en algoritmos de clasificacion no hay que estandarizarla porque yo lo que quiero es que me diga si compra o no. En el caso de algoritmos de regresion por ejemplo,
##si habria que escalarlas.

##* En R las variables que hemos convertido en numerica que son categoricas, no se pueden escalar asi que escalamos solamente las numericas
