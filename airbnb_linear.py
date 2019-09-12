import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

raw_data = pd.read_csv("AB_NYC_2019.csv")
df = raw_data[(raw_data.price <= 600) | (raw_data.availability_365 != 0)].copy()

####################
# pre-process data #
####################

# reviews_per_month: replace null with average value
avg_reviews_per_month = df["reviews_per_month"].mean()
df["reviews_per_month"] = df["reviews_per_month"].fillna(avg_reviews_per_month)

# last_review data: convert it to numeric value
df["last_review"] = pd.to_datetime(df["last_review"], infer_datetime_format=True)
avg_last_review = df["last_review"].mean()
df["last_review"] = df["last_review"].fillna(avg_last_review)
df["last_review"] = df["last_review"].apply(lambda review_date: review_date.toordinal() - avg_last_review.toordinal())

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

# split data set to training data and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=50)

# normalize features in order to better train our model
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

############################
# train and evaluate model #
############################

model = LinearRegression()
model.fit(x_train_scaled, y_train)
print(model.score(x_test_scaled, y_test))
