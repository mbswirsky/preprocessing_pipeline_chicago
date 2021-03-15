import numpy as np
import pandas as pd
import pickle

df = pd.read_csv("data/traffic_crashes_chicago.csv")

# Make some sample "new" data
df_new = df.sample(30000)
X_new = df_new.drop(columns="DAMAGE")

with open('pipe_forest.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

preds = loaded_model.predict(X_new)

print(f'{len(preds)} prediction labels.')
print(pd.Series(preds).value_counts().to_string())
