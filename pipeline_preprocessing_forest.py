# import warnings
# warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import time

from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

from chicago_preprocessor.utils import column_separator, sample_splitter
from chicago_preprocessor.transformers import (
    NullMaker,
    BeatFormatter,
    DateFormatter,
    DateEncoder,
)


df = pd.read_csv("data/traffic_crashes_chicago.csv")

# Separate out columns to be treated differently

# Outcome variable
target_col = "DAMAGE"

# Beat of occurrence
beat_cols = ["BEAT_OF_OCCURRENCE"]

# Specify date columns specifically
date_cols = ["CRASH_DATE", "DATE_POLICE_NOTIFIED"]

# Triage data types
# Remove columns with more than 10% null values or with more than 100 categories
null_cols, num_cols, cat_cols = column_separator(
    df,
    target_col,
    beat_cols,
    date_cols,
    null_thresh=0.1,
    cat_thresh=100,
)
print("Columns identified")

# Simplify target variable
df[target_col] = df[target_col].where(df[target_col] == "OVER $1,500", "$1,500 OR LESS")

# Split out training and test sets
X_train, X_test, y_train, y_test = sample_splitter(
    df, target_col, subsample_size=10000, test_size=0.1
)

print("Train/test splitted")

# Set up pipelines
beat_transformer = Pipeline(
    steps=[
        ("nuller", NullMaker()),
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("formatter", BeatFormatter()),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ("scaler", StandardScaler()),
    ]
)

date_transformer = Pipeline(
    steps=[
        ("nuller", NullMaker()),
        ("formatter", DateFormatter()),
        ("encoder", DateEncoder()),
        ("scaler", StandardScaler()),
    ]
)

num_transformer = Pipeline(
    steps=[
        ("nuller", NullMaker()),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

cat_transformer = Pipeline(
    steps=[
        ("nuller", NullMaker()),
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ("scaler", StandardScaler()),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("beat", beat_transformer, beat_cols),
        ("date", date_transformer, date_cols),
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols),
    ]
)

pca = PCA(n_components=50)

rfc = RandomForestClassifier(max_depth=10)

pipe = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("pca", pca),
        ("clf", rfc),
    ]
)

print("Pipeline defined")

# Fit model
t0 = time.perf_counter()
pipe.fit(X_train, y_train)
print("Model fitted")
print("Execution time: {:.2f} seconds\n".format(time.perf_counter() - t0))

# Make predictions
y_train_pred = pipe.predict(X_train)
y_pred = pipe.predict(X_test)
print("Train:\n", classification_report(y_train, y_train_pred), "\n\n")
print("Test:\n", classification_report(y_test, y_pred), "\n")

# Save results
# with open('results.txt', 'w') as f:
#     print(results, file=f)

# Save model
with open("pipe_forest.pkl", "wb") as f:
    pickle.dump(pipe, f)

print("Pickled and done")
