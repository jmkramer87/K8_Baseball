import pandas as pd
import numpy as np

#Data Preparation
df = pd.read_csv('Batters_22-24_Avg_Clean_Nameless.csv', encoding='UTF-8')
writePath = 'logs-jkd22.txt'

avg = df.mean()

with open(writePath, 'a') as f:
    f.write("Starting Analysis\n")
    f.write(avg.to_string(header=False, index=False))
    f.write("Job complete!")