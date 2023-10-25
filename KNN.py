"""
Created on Tue Oct 24 23:27:27 2023

@author: Bren Guzmán, Brenda García, María José Merino

"""

import numpy as np

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def euclidean_distance(self, x1, x2):
        # Calcula la distancia euclidiana entre dos vectores x1 y x2
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            # Calcula las distancias entre x y todos los puntos en el conjunto de entrenamiento
            distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
            # Obtiene los índices de los k puntos más cercanos
            k_indices = np.argsort(distances)[:self.k]
            # Obtiene las etiquetas de los k puntos más cercanos
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            # Elige la etiqueta más común entre los k puntos más cercanos
            most_common = np.bincount(k_nearest_labels).argmax()
            y_pred.append(most_common)
        return np.array(y_pred)

    def accuracy(self, y_true, y_pred):
        # Calcula la precisión (accuracy) de las predicciones
        correct = np.sum(y_true == y_pred)
        return correct / len(y_true)
