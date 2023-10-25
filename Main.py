"""
Created on Wed Oct 25 00:08:27 2023

@author: Bren Guzmán
"""

#%% LIBRERÍAS
from Dataframe import Dataframe
from KNN import KNN

#%% CARGAR ARCHIVO
dataframe = Dataframe()
dataframe.load('data.csv')  
X_train, X_test, y_train, y_test = dataframe.split_data(0.2)  

df = dataframe.data()

#%% CLASIFICAR


knn = KNN(k=3)

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = knn.accuracy(y_test, y_pred)
print(f'Accuracy: {accuracy}')
