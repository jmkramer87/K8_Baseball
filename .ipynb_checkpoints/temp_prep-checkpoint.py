import pandas as pd
import numpy as np

#Data Preparation
df = pd.read_csv('Batters_22-24_Avg_Clean_Nameless.csv', encoding='UTF-8')

f = open("logs-jkd22.txt", "w")

f.write("Starting Analysis\n")

f.write(df.head())

f.write(df.mean())

f.write("Job complete!")

f.close()
