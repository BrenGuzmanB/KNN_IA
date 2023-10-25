"""
Created on Sat Oct 21 01:12:10 2023

@author: Bren Guzmán
"""

import numpy as np

class Dataframe(object):
    def __init__(self):
        # Public
        self.numSample = None  # Se inicializará después de cargar los datos
        self.numAttrib = None  # Se inicializará después de cargar los datos
        # Private
        self.__data = []       # Los datos se cargarán desde un archivo
        self.__label = []      # Las etiquetas se cargarán desde un archivo

    def data(self):
        return self.__data

    def label(self):
        return self.__label

    def split_data(self, test_size):
        # Realiza una división de los datos en conjuntos de entrenamiento y prueba
        if self.numSample is None or self.numAttrib is None:
            raise ValueError("Número de muestras y atributos no definidos. Cargue los datos primero.")
        
        num_samples = self.numSample
        num_attributes = self.numAttrib
        num_test_samples = int(num_samples * test_size)

        # Realiza una permutación de los índices para dividir los datos en forma aleatoria
        indices = np.random.permutation(num_samples)

        # Divide los índices en conjuntos de entrenamiento y prueba
        train_indices = indices[num_test_samples:]
        test_indices = indices[:num_test_samples]

        # Divide los datos y etiquetas en conjuntos de entrenamiento y prueba
        X_train = [self.__data[i] for i in train_indices]
        y_train = [self.__label[i] for i in train_indices]
        X_test = [self.__data[i] for i in test_indices]
        y_test = [self.__label[i] for i in test_indices]

        return X_train, X_test, y_train, y_test

    # Función para leer datos de un archivo
    def load(self, Name):
        try:
            data = open(Name, 'r')
        except:
            print('El archivo no puede abrirse')
            quit()

        mat = []
        for linea in data:
            linea = linea.rstrip()
            linea = linea.split(',')
            mat.append([float(x) for x in linea])

        data.close()

        # Determina el número de muestras y atributos a partir de los datos cargados
        self.numSample = len(mat)
        self.numAttrib = len(mat[0])

        # Convierte los datos y etiquetas a NumPy arrays
        self.__data = np.array([line for line in mat])
        self.__label = np.array([line[-1] for line in mat])



#%% Ejemplo de uso:
    '''
dataframe = Dataframe()
dataframe.load('data.csv')  
X_train, X_test, y_train, y_test = dataframe.split_data(0.2)  

df = dataframe.data()

'''

    


