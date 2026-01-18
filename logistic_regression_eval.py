import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error)
import numpy as np
import pandas as pd

model = joblib.load('logistic_regression_model.pkl')

data_path = r'C:.\highD\data\01_tracks.csv'

df = pd.read_csv(data_path)

df = df[(df['ttc'] > 0) & (df['ttc'] <= 10)]

X = df[['dhw', 'xVelocity', 'yVelocity', 'width', 'height', 'precedingXVelocity', 'xAcceleration', 'yAcceleration']]
Y = (df['ttc'] < 2).astype(int)
X_np = X.to_numpy()
Y_np = Y.to_numpy()
Y_pred = model.predict(X)
# --- 1. Klassifikations-Metriken (Gefahrenerkennung) ---
accuracy = accuracy_score(Y_np, Y_pred)
precision = precision_score(Y_np, Y_pred)
recall = recall_score(Y_np, Y_pred)
f1 = f1_score(Y_np, Y_pred)

mse = mean_squared_error(Y_np, Y_pred)
rmse = np.sqrt(mse)
correlation = np.corrcoef(Y_np.flatten(), Y_pred.flatten())[0, 1]
std_dev = np.std(Y_np - Y_pred) # Standardabweichung der Fehler

print("\n--- Ergebnisse der Logistischen Regression ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f} (Vermeidung von Fehlalarmen)")
print(f"Recall: {recall:.4f} (Erkennungsrate echter Gefahren)")
print(f"F1-Score: {f1:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"Korrelation: {correlation:.4f}")
print(f"Standardabweichung (Fehler): {std_dev:.4f}")

importances = model.coef_[0]
feature_names = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.plot(Y.values, label='Tatsächliche Kritikalität (Ground Truth)', color='blue', linewidth=2)
plt.plot(Y_pred, label= 'Logistische Regression Vorhersage', color='orange', linestyle='--')
plt.axhline(y=0.5, color='red', label='Kritischer Schwellenwert (0.5)', linestyle=':')
plt.title('Logistische Regression Kritikalitätsvorhersage')
plt.xlabel('Probenindex')
plt.ylabel('Kritikalität (1 = Kritisch, 0 = Sicher)')
plt.legend()

plt.figure(figsize=(10, 6))
plt.title('Feature-Wichtigkeit der Logistischen Regression')
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), feature_names[indices], rotation=45)
plt.xlabel('Feature')
plt.ylabel('Koeffizient')
plt.grid()


plt.show()