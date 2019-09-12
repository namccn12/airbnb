import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the data into a DataFrame
df = pd.read_csv('C:\\Users\\798001\\Documents\\NYC AirBnB\\AB_NYC_2019.csv', header=0)

# Replace Na's in last review and reviews_per_month with 0s
df["last_review"].fillna(0, inplace=True)
df["reviews_per_month"].fillna(0, inplace=True)

# Set index for any loc lookups
df.set_index("id", inplace=True)

# filter out price over $600
outdf = df[df.price < 600]

# filter out columns with no reviews
zer = df[df.number_of_reviews == 0]
print("percentage of zero reviews " + len(zer)/len(outdf))

# filter out only specific neighbourhood groups
manhat = outdf[outdf.neighbourhood_group == "Manhattan"]
brook = outdf[outdf.neighbourhood_group == "Brooklyn"]

# filter min nights less than 50 nights
minnights = outdf[outdf.minimum_nights < 50]

# display distribution of min nights after filter
sns.set()
sns.distplot(minnights["minimum_nights"])
plt.show()

print("percentage of price over 500 " + len(df[df.price > 500])/len(df))