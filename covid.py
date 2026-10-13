
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("country_wise_latest.csv")
print(dataset.head())
print(dataset.info())
print(dataset.isnull().sum())
print("Duplicated rows:", dataset.duplicated().sum())

dataset.drop_duplicates(inplace=True)
dataset.replace([np.inf, -np.inf], np.nan, inplace=True)

dataset.fillna(dataset.mean(numeric_only=True), inplace=True)

for col in dataset.select_dtypes(include=["object"]).columns:
    dataset[col] = dataset[col].fillna(dataset[col].mode()[0])

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
dataset["Country/Region"] = le.fit_transform(dataset["Country/Region"])
print(dataset["Country/Region"])
print(le.classes_)

le = LabelEncoder()
dataset["WHO Region"] = le.fit_transform(dataset["WHO Region"])
print(dataset["WHO Region"])
print(le.classes_)

X = dataset.drop(columns=["Deaths"]).values
Y = dataset["Deaths"].values

X_df = pd.DataFrame(
    X,
    columns=dataset.drop(columns=["Deaths"]).columns
)
print(X_df.describe())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=1
)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


X_original = X.copy()

X_train_df = pd.DataFrame(
    X_train,
    columns=X_df.columns
)

print("BEFORE SCALING")
print(pd.DataFrame(
    X_original,
    columns=X_df.columns
).describe().round(2))

print("\nAFTER SCALING (TRAINING DATA)")
print(X_train_df.describe().round(2))

