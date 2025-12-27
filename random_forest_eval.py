import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

model = joblib.load('ttc_random_forest_model.pkl')

data_path = r'C:.\highD\data\01_tracks.csv'

df = pd.read_csv(data_path)

df = df[(df['ttc'] > 0) & (df['ttc'] <= 10)]

X = df[['dhw', 'xVelocity', 'yVelocity', 'width', 'height', 'precedingXVelocity', 'xAcceleration', 'yAcceleration']]
Y = df ['ttc']

Y_pred = model.predict(X)

r_sq = r2_score(Y, Y_pred)
rmse = np.sqrt(mean_squared_error(Y, Y_pred))

print("--- Ergebnisse vom Random Forest ---")
print(f"R-Quadrat-Wert (R²): {r_sq:.4f}")
print(f"RMSE (Mittlerer Fehler in Sek.): {rmse:.4f}")

importances = model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))

plt.title('Welche Faktoren beinflussen die Unfallwahrscheinlichkeit (TTC)?')
plt.bar(range(X.shape[1]), importances[indices], align='center', color='skyblue')
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45)
plt.ylabel('Wichtigkeit (Gini Importance)')
plt.tight_layout()
plt.show()