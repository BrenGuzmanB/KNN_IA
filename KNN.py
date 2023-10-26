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

    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            # Calcula las distancias entre x y todos los puntos en el conjunto de entrenamiento
            distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
            # Obtiene los índices de los k puntos más cercanos
            k_indices = np.argsort(distances)[:self.k]
            # Obtiene las etiquetas de los k puntos más cercanos
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            
            # Imprimir información sobre el punto y sus k vecinos cercanos
            print(f"\n\nPoint to Predict: {x}")
            for i, idx in enumerate(k_indices):
                neighbor = self.X_train[idx]
                neighbor_label = k_nearest_labels[i]
                distance = distances[idx]
                print(f"Neighbor {i + 1}: {neighbor}, Label: {neighbor_label}, Distance: {distance}")
        
            # Calcula la etiqueta más común entre los k puntos más cercanos
            counts = {}
            for label in k_nearest_labels:
                if label in counts:
                    counts[label] += 1
                else:
                    counts[label] = 1
            most_common = max(counts, key=counts.get)
            
            y_pred.append(most_common)
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


  