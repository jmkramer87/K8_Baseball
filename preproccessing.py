import pandas as pd
import numpy as np
import math
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVR

#Data Preparation

def whipCreator(df):
    #Takes dataframe and creates whip values

    outsCol = []
    for p in df['p_formatted_ip']:
        if(p - math.floor(p) == 0.1):
            temp = math.floor(p) + 0.333333
        elif(p - math.floor(p) == 0.2):
            temp = math.floor(p) + 0.666667
        else:
            temp = p
        outsCol.append(temp)
    df['ip'] = outsCol
    df['whip'] = (df['walk']+df['hit'])/df['ip']

def cleaner(df):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    df = df.select_dtypes(include=numerics)
    #whipCreator(df)

    return df

def name(df1, df2, df3):
    df1 = df1[['last_name, first_name', 'player_id']]
    df2 = df2[['last_name, first_name', 'player_id']]
    df3 = df3[['last_name, first_name', 'player_id']]

    names = df1.merge(df2, on='player_id')
    names = names.merge(df3, on='player_id')

    return names

df1 = pd.read_csv('~\\Downloads\\Batters 2021.csv', encoding='UTF-8')
df2 = pd.read_csv('~\\Downloads\\Batters 2022.csv', encoding='UTF-8')
df3 = pd.read_csv('~\\Downloads\\Batters 2023.csv', encoding='UTF-8')

temp = name(df1,df2,df3)

df1 = cleaner(df1)
df2 = cleaner(df2)
df3 = cleaner(df3)

df = pd.concat((df1, df2, df3))
df = df.groupby('player_id').mean()
df = df.dropna(axis=1)
#df = df.drop(columns='p_formatted_ip')

df = df.merge(temp, on='player_id')
df = df.drop(columns=['last_name, first_name_x', 'last_name, first_name_y', 'year'])

names = df['last_name, first_name']
df.drop(labels=['last_name, first_name'], axis=1,inplace = True)
df.insert(0, 'last_name, first_name', names)

df.to_csv('~\\Downloads\\Batters 21-23 Average Cleaned.csv', encoding='utf-8')
