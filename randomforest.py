import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

data_path_folder = r'C:.\highD\data'

data_set_file = r'C:.\highD\data\35_tracks.csv'

df = pd.read_csv(data_set_file)

df = df[(df['ttc'] > 0) & (df['ttc'] <= 10)]

#df = df[df['dhw'] > 0]

X = df[['dhw', 'xVelocity', 'yVelocity', 'width', 'height', 'precedingXVelocity', 'xAcceleration', 'yAcceleration']]
Y = df ['ttc']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

r_sq = r2_score(Y_test, Y_pred)
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))

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