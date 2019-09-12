import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("AB_NYC_2019.csv")

print(df.isnull().sum())
