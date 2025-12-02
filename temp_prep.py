import pandas as pd
import numpy as np

#Data Preparation
df = pd.read_csv('~/data/Batters 22-24 Average Cleaned.csv', encoding='UTF-8')

f = open("logs-jkd22.txt", "a")

f.write("Starting Analysis\n")

f.write(df.head())

f.write(df.mean())

f.write("Job complete!")

f.close()