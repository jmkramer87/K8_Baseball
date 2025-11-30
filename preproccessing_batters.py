import pandas as pd
import numpy as np
import math

#Data Preparation

def cleaner(df):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    df = df.select_dtypes(include=numerics)

    df = df.dropna(axis=1)

    return df

def name(df1, df2, df3):
    df1 = df1[['name', 'player_id']]
    df2 = df2[['name', 'player_id']]
    df3 = df3[['name', 'player_id']]

    names = df1.merge(df2, on='player_id')
    names = names.merge(df3, on='player_id')

    return names

df1 = pd.read_csv('~/data/Batters 2022a.csv', delimiter=';', encoding='UTF-8')
df2 = pd.read_csv('~/data/Batters 2023a.csv', delimiter=';', encoding='UTF-8')
df3 = pd.read_csv('~/data/Batters 2024a.csv', delimiter=';', encoding='UTF-8')

temp = name(df1,df2,df3)

df1 = cleaner(df1)
df2 = cleaner(df2)
df3 = cleaner(df3)

df = pd.concat((df1, df2, df3))
df = df.groupby('player_id').mean()
df = df.dropna(axis=1)

df = df.merge(temp, on='player_id')
df = df.drop(columns=['name_x', 'name_y', 'year'])

names = df['name']
df.drop(labels=['name'], axis=1,inplace = True)
df.insert(0, 'name', names)

#df.to_csv('~/data/Batters 22-24 Average Cleaned.csv', encoding='utf-8')
