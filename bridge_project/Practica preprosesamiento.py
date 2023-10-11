# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 16:38:03 2023

@author: lenovo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('chess_games.csv')


data.info()
data.describe()
data.head()

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
    # List of unique values
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

#%% Obtaining the data quality report
report = dqr(data)

#%% DATA PROCESSING

#rellenando la columna Opening_variation los blanks con "traditional opening"

data['opening_variation'] = data['opening_variation'].fillna('traditional opening')

#drop de columnas no significativas basado en el conocimiento de Chess y el contenido de la variable
#'white_id','black_id','moves'

#drop de la columna 'opening_response' tiene 93% de sus datos vacíos
data = data.drop(labels=['white_id','black_id','moves','opening_response'],axis=1)
data.head()
data.info()
df = data

# Codificando todas las variables categóricas consideradas
# Initialize a LabelEncoder for each categorical column
le1 = LabelEncoder() #victory_status
le2 = LabelEncoder() #winner
le3 = LabelEncoder() #time_increment
le4 = LabelEncoder() #opening_code
le5 = LabelEncoder() #opening_fullname
le6 = LabelEncoder() #opening_shortname
le7 = LabelEncoder() #opening_variation
le8 = LabelEncoder() #rated

# Apply label encoding to each categorical column
df['victory_status_cod'] = le1.fit_transform(df['victory_status'])
df['winner_cod'] = le2.fit_transform(df['winner'])
df['time_increment_cod'] = le3.fit_transform(df['time_increment'])
df['opening_code_cod'] = le4.fit_transform(df['opening_code'])
df['opening_fullname_cod'] = le5.fit_transform(df['opening_fullname'])
df['opening_shortname_cod'] = le6.fit_transform(df['opening_shortname'])
df['opening_variation_cod'] = le7.fit_transform(df['opening_variation'])
df['rated_cod'] = le7.fit_transform(df['rated'])

#%% RESAMPLE DE 20000 A 2000 MUESTRAS



#%% GENERANDO DATASETS CON DATOS ORIGINALES (CODIFICADOS)

df1 = df.drop(labels=['victory_status','winner','time_increment','opening_code','opening_fullname','opening_shortname','opening_variation','rated'],axis=1)
df2 = df1.drop(labels=['opening_fullname_cod'],axis=1)
df3 = df1.drop(labels=['opening_shortname_cod','opening_variation_cod'],axis=1)




df1.info()

# Export the DataFrame to a CSV file
df1.to_csv('df1.csv', index=False)
df2.to_csv('df2.csv', index=False)
df3.to_csv('df3.csv', index=False)



sns.pairplot(df1)


#%% RANDOMLY RESAMPLE TO 2000 SAMPLES ONLY
df_2000 = df.sample(n=2000, random_state=42)

df1_2000 = df_2000.drop(labels=['victory_status','winner','time_increment','opening_code','opening_fullname','opening_shortname','opening_variation','rated'],axis=1)
df2_2000 = df1_2000.drop(labels=['opening_fullname_cod'],axis=1)
df3_2000 = df1_2000.drop(labels=['opening_shortname_cod','opening_variation_cod'],axis=1)

df1_2000.to_csv('df1_2000.csv', index=False)
df2_2000.to_csv('df2_2000.csv', index=False)
df3_2000.to_csv('df3_2000.csv', index=False)

sns. pairplot(df1_2000)


#%%  IDENTIFICATION OF ATYPICAL VALUES 
#we consider only original cuantitative variables for BOX PLOT
df1_2000 = df1_2000.drop(labels=['game_id'],axis=1)
df1_2000_num = df1_2000.drop(labels=['victory_status_cod','winner_cod','time_increment_cod','opening_code_cod','opening_fullname_cod','opening_shortname_cod','opening_variation_cod','rated_cod'],axis=1)
boxplot = df1_2000_num.boxplot(fontsize=9, figsize=(8,5), rot=45)

#%% Function to determine the outliers
def find_boundaries(df_var,distance=1.5):
    IQR = df_var.quantile(0.75)-df_var.quantile(0.25)
    lower = df_var.quantile(0.25)-IQR*distance
    upper = df_var.quantile(0.75)+IQR*distance
    return lower,upper

# me da una lista con los outliers de la variable turns
lmin,lmax = find_boundaries(df1_2000['turns'])
outliers = np.where(df1_2000['turns'] > lmax, True,np.where(df1_2000['turns'] < lmin, True, False))
outliers_df = df1_2000.loc[outliers, 'turns']
indexes_turns = outliers_df.index
index_list_turns = indexes_turns.tolist()

# me da un lista con los outliers de la variable white_rating
lmin,lmax = find_boundaries(df1_2000['white_rating'])
outliers = np.where(df1_2000['white_rating'] > lmax, True,np.where(df1_2000['white_rating'] < lmin, True, False))
outliers_df = df1_2000.loc[outliers, 'white_rating']
indexes_white_rating = outliers_df.index
index_list_white_rating = indexes_white_rating.tolist()

# me da un lista con los outliers de la variable black_raiting
lmin,lmax = find_boundaries(df1_2000['black_rating'])
outliers = np.where(df1_2000['black_rating'] > lmax, True,np.where(df1_2000['black_rating'] < lmin, True, False))
outliers_df = df1_2000.loc[outliers, 'black_rating']
indexes_black_rating = outliers_df.index
index_list_black_rating = indexes_black_rating.tolist()

# me da un lista con los outliers de la variable opening_moves
lmin,lmax = find_boundaries(df1_2000['opening_moves'])
outliers = np.where(df1_2000['opening_moves'] > lmax, True,np.where(df1_2000['opening_moves'] < lmin, True, False))
outliers_df = df1_2000.loc[outliers, 'opening_moves']
indexes_opening_moves = outliers_df.index
index_list_opening_moves = indexes_opening_moves.tolist()

# Concatenate the lists
concatenated_index_lists = index_list_turns + index_list_white_rating + index_list_black_rating + index_list_opening_moves

# Convert the concatenated list into a set to remove duplicates
unique_indexes = list(set(concatenated_index_lists))

#%% REMOVING OUTLIERS

df_cleaned = df1_2000.drop(unique_indexes)

df_cleaned.to_csv('df1_2000_cleanned.csv', index=False)

#%% ASYMMETRY IN THE DATA

# Calculation of skewness with pandas
skewness = df_cleaned.skew()

#%% Correlation analysis for elimination of variables
df_cleaned.info()
# Calculate the correlation matrix
correlation_matrix = df_cleaned.corr()
# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


#%% Hierarchical clustering application
from scipy.cluster.hierarchy import dendrogram, linkage
Z = linkage(df_cleaned.T,metric='correlation',method='complete')

d = dendrogram(Z)
plt.show()
correlaciones_clust = np.corrcoef(df_cleaned.iloc[:,d['winner_cod']],rowvar=False)
fig =  plt.figure()
plt.imshow(correlaciones_clust)
plt.xticks(np.arange(10),d['winner_cod'])
plt.yticks(np.arange(10),d['winner_cod'])
plt.colorbar()
plt.show()
# fig.savefig('../figures/P1_fig/F11.png')


