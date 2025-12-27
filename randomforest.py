import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import joblib

data_path_folder = r'C:.\highD\data'
search_pattern = os.path.join(data_path_folder, '*_tracks.csv')
all_files = glob.glob(search_pattern)

all_dfs = []

for file_path in all_files:
    base_name = os.path.basename(file_path)
    recording_id = base_name.split('_')[0]

    df_tmp = pd.read_csv(file_path, usecols=['id', 'ttc', 'dhw', 'xVelocity', 'yVelocity', 'width', 'height', 'precedingXVelocity', 'xAcceleration', 'yAcceleration'])
    df_tmp['id'] = recording_id + '_' + df_tmp['id'].astype(str)

    df_tmp = df_tmp[(df_tmp['ttc'] > 0) & (df_tmp['ttc'] <= 10)]

    all_dfs.append(df_tmp)

df = pd.concat(all_dfs, ignore_index=True)

X = df[['dhw', 'xVelocity', 'yVelocity', 'width', 'height', 'precedingXVelocity', 'xAcceleration', 'yAcceleration']]
Y = df ['ttc']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

model.fit(X_train, Y_train)

joblib.dump(model, 'ttc_random_forest_model.pkl')
print("Random Forest Modell gespeichert")