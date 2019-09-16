import pandas as pd
import seaborn
import matplotlib.pyplot as plt

data = pd.read_csv("AB_NYC_2019.csv")
df = data[(data.price <= 600) & (data.neighbourhood_group == "Brooklyn")].copy()

seaborn.distplot(df.number_of_reviews)
plt.show()
