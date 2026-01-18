import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import joblib
import glob
import os
import utils.save_split as save_split

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

features = ['dhw', 'xVelocity', 'yVelocity', 'width', 'height', 'precedingXVelocity', 'xAcceleration', 'yAcceleration']

df['is_critical'] = (df['ttc'] < 2).astype(int)

X = df[features]
Y = df['is_critical']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

save_split.save_processed_split(X_train, X_test, Y_train, Y_test, scaler, 'data/logistic_regression')

model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

print("\n--- Ergebnisse Logistische Regression ---")
print(f"Accuracy:  {accuracy_score(Y_test, Y_pred):.4f}")
print(f"Precision: {precision_score(Y_test, Y_pred):.4f}")
print(f"Recall:    {recall_score(Y_test, Y_pred):.4f}")
print(f"F1-Score:  {f1_score(Y_test, Y_pred):.4f}")

print("\nKonfusionsmatrix:")
print(confusion_matrix(Y_test, Y_pred))

# 8. Feature-Gewichtung (Was beeinflusst die Entscheidung?)
importance = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

print("\nFeature Gewichte (Koeffizienten):")
print(importance)

# 9. Speichern
joblib.dump(model, 'data/logistic_regression/logistic_regression_model.pkl')
print("\nModell wurde gespeichert.")