import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score

raw_data = pd.read_csv("AB_NYC_2019.csv")
df = raw_data[raw_data.price <= 600].copy()

####################
# pre-process data #
####################

# reviews_per_month: replace null with 0
df["reviews_per_month"] = df["reviews_per_month"].fillna(0)

# last_review data: convert it to numeric value
df["last_review"] = pd.to_datetime(df["last_review"], infer_datetime_format=True)
earliest_last_review = min(df["last_review"])
df["last_review"] = df["last_review"].fillna(earliest_last_review)
df["last_review"] = df["last_review"].apply(
    lambda review_date: review_date.toordinal() - earliest_last_review.toordinal())

# neighbourhood: label encoding
neighbourhood_encoder = LabelEncoder()
neighbourhood_labels = neighbourhood_encoder.fit_transform(df["neighbourhood"])
df["neighbourhood"] = neighbourhood_labels
# retain the mapping of neighbourhood and encoded values
# neighbourhood_dict = dict(zip(neighbourhood_encoder.classes_, range(len(neighbourhood_encoder.classes_))))

# room_type: label encoding
room_encoder = LabelEncoder()
room_labels = room_encoder.fit_transform(df["room_type"])
df["room_type"] = room_labels
# retain the mapping of room_type and encoded values
# room_dict = dict(zip(room_encoder.classes_, range(len(room_encoder.classes_))))

# convert feature to log(1 + feature)
df["price"] = np.log1p(df["price"])

#######################
# select key features #
#######################

x = df[["neighbourhood", "latitude", "longitude", "room_type", "minimum_nights", "number_of_reviews"]]

y = list(df["price"])

# normalize features in order to better train our model
scaler = MinMaxScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)

###############################
# train model and plot result #
###############################

k_scores = []
k_range = range(20, 60)

for k in k_range:
    model = KNeighborsRegressor(n_neighbors=k, weights='distance')
    scores = cross_val_score(model, x_scaled, y, cv=5)
    k_scores.append(np.mean(scores))

plt.plot(k_range, k_scores)
plt.title("R2_score (knn with cross validation)")
plt.xlabel("k")
plt.show()
