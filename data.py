# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 23:30:32 2023

@author: Bren Guzmán
"""

import pandas as pd



# Cargar los archivos CSV en DataFrames
train_df = pd.read_csv('train.tra')
test_df = pd.read_csv('test.tes')

# Asignar nombres temporales a las columnas
column_names = [f'atributo{col}' for col in range(1, 65)] + ['etiqueta']
train_df.columns = column_names
test_df.columns = column_names

# Concatenar los DataFrames en uno solo
combined_df = pd.concat([train_df, test_df], ignore_index=True)

# Guardar el DataFrame combinado en un archivo CSV sin nombres de columna
combined_df.to_csv('data.csv', index=False, header=False)
