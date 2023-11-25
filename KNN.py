"""
Created on Tue Oct 24 23:27:27 2023

@author: Bren Guzmán, Brenda García, María José Merino

"""

import numpy as np
import matplotlib.pyplot as plt
import random

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.unique_labels = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.unique_labels = self.get_unique_labels(y_train)

    def euclidean_distance(self, x1, x2):        
        if len(x1) != len(x2):
            raise ValueError("Los vectores deben tener la misma longitud")
    
        squared_distance = 0
        for i in range(len(x1)):
            squared_distance += (x1[i] - x2[i]) ** 2
    
        distance = squared_distance ** 0.5
        return distance

    def predict(self, X_test, y_test, show_misclassified=False):
        y_pred = []
        misclassified = []

        for i, x in enumerate(X_test):
            # Calcula las distancias entre x y todos los puntos en el conjunto de entrenamiento
            distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
            # Obtiene los índices de los k puntos más cercanos
            k_indices = np.argsort(distances)[:self.k]
            # Obtiene las etiquetas de los k puntos más cercanos
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            
            # Calcula la etiqueta más común entre los k puntos más cercanos
            counts = {}
            for label in k_nearest_labels:
                if label in counts:
                    counts[label] += 1
                else:
                    counts[label] = 1
            most_common = max(counts, key=counts.get)
            
            y_pred.append(most_common)

            # Verifica si la predicción es incorrecta
            if most_common != y_test[i]:
                misclassified.append((i, y_test[i], most_common))
        
        if show_misclassified:
            self.show_misclassified_images(X_test, misclassified)

        return np.array(y_pred)

    def accuracy(self, y_true, y_pred):
        # Calcula la precisión (accuracy) de las predicciones
        correct = np.sum(y_true == y_pred)
        return correct / len(y_true)

    def get_unique_labels(self, labels):
        unique_labels = set()
        for label in labels:
            unique_labels.add(label)
        return list(unique_labels)
    
    def confusion_matrix(self, y_true, y_pred):
        if self.unique_labels is None:
            raise ValueError("Entrena el modelo utilizando el método 'fit' antes de calcular la matriz de confusión.")
        
        num_labels = len(self.unique_labels)
        matrix = np.zeros((num_labels, num_labels), dtype=int)
        
        label_to_index = {label: i for i, label in enumerate(self.unique_labels)}
        
        for true, pred in zip(y_true, y_pred):
            true_index = label_to_index[true]
            pred_index = label_to_index[pred]
            matrix[true_index][pred_index] += 1
        
        return matrix
    
    def show_misclassified_images(self, X_test, misclassified):
        # Muestra los dígitos clasificados incorrectamente
        for index, true_label, pred_label in misclassified:
            plt.figure()
            plt.imshow(X_test[index].reshape(8, 8), cmap='gray', interpolation='nearest')
            plt.title(f"True Label: {true_label}, Predicted Label: {pred_label}")
            plt.show()
            
    def plot_random_images(self, X_test, y_pred, n=5):
        # Selecciona n índices aleatorios de las imágenes de prueba
        random_indices = random.sample(range(len(X_test)), n)

        # Muestra las imágenes seleccionadas aleatoriamente
        for index in random_indices:
            plt.figure()
            plt.imshow(X_test[index].reshape(8, 8), cmap='gray', interpolation='nearest')
            plt.title(f"Predicted Label: {y_pred[index]}")
            plt.show()
  