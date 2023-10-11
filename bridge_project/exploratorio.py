# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 01:29:49 2023

@author: L00739961
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_squared_error,r2_score)
from sklearn.preprocessing import MinMaxScaler



#%% Cargar los datos para procesarlos
df=pd.read_csv("C:/Users/L00739961/Desktop/Respaldo febrero 15 2022/ITS/Modelado predictivo/Proyecto 1/chess_games.csv")
df.head()
#%% DATA QUALITY REPORT
def dqr(data):
    # List of database variables
    cols = pd.DataFrame(list(data.columns.values),
                           columns=['Names'],
                           index=list(data.columns.values))
    # List of data types
    dtyp = pd.DataFrame(data.dtypes,columns=['Type'])
    # List of missing data
    misval = pd.DataFrame(data.isnull().sum(),
                                  columns=['Missing_values'])
    # List of present data
    presval = pd.DataFrame(data.count(),
                                  columns=['Present_values'])
    # List of unique values, cuenta el número de valores únicos en la columna
    unival = pd.DataFrame(columns=['Unique_values'])
    # List of min values
    minval = pd.DataFrame(columns=['Min_value'])
    # List of max values
    maxval = pd.DataFrame(columns=['Max_value'])
    for col in list(data.columns.values):
        unival.loc[col] = [data[col].nunique()]
        try:
            minval.loc[col] = [data[col].min()]
            maxval.loc[col] = [data[col].max()]
        except:
            pass
    
    # Join the tables and return the result
    return cols.join(dtyp).join(misval).join(presval).join(unival).join(minval).join(maxval)
#%% Obtaining the data quality report, ejecuta el bloquecito de arriba con los datos actuales
#cambiar el nombre entre paréntesis por el nombre del archivo.
report = dqr(df)
#%% Valores de las variables y número de veces que aparecen
val_rated=pd.value_counts(df.rated)
val_rated
val_winner=pd.value_counts(df.winner)
val_winner
val_victory_status=pd.value_counts(df.victory_status)
val_victory_status
val_white_id=pd.value_counts(df.white_id)
val_white_id
val_black_id=pd.value_counts(df.black_id)
val_black_id
val_opening_code=pd.value_counts(df.opening_code)
val_opening_code
val_opening_fullname=pd.value_counts(df.opening_fullname)
val_opening_fullname
val_opening_shortname=pd.value_counts(df.opening_shortname)
val_opening_shortname
val_opening_response=pd.value_counts(df.opening_response)
val_opening_response
val_opening_variation=pd.value_counts(df.opening_variation)
val_opening_variation


#%%Gráfica de turnos
fig, ax = plt.subplots(figsize=(10,7))
sns.histplot(x="turns", hue="winner", data=df, multiple='stack', linewidth=0.5, edgecolor=".3")
plt.savefig('turns.png')

#%% Histrogramas
df_x=df.drop(["winner","game_id", "rated"], axis="columns")

# Imprimir la distribución de las variables
# Se define el grid de la subplot
#subplots me da varias gráficas en un mismo arreglo, en este caso de tamaño 4x4
#figsize determina el tamaño del gráfico en su totalidad
fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(15, 15))
#hspace=0.5 es la distacia entre cada pequeña gráfica
plt.subplots_adjust(hspace=0.5)
fig.suptitle("Distribución de las variables", fontsize=18, y=0.95)
#axs.ravel entrega un array de una sola dimension
for col_name, ax in zip(df_x.columns, axs.ravel()):
    ax.hist(df_x[col_name].dropna(), bins=10)
    ax.set_title(col_name)
plt.savefig("Histogramas.png")
