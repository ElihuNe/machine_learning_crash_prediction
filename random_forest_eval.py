import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error)
import numpy as np
import pandas as pd

model = joblib.load('ttc_random_forest_model.pkl')

data_path = r'C:.\highD\data\01_tracks.csv'

df = pd.read_csv(data_path)

df = df[(df['ttc'] > 0) & (df['ttc'] <= 10)]

X = df[['dhw', 'xVelocity', 'yVelocity', 'width', 'height', 'precedingXVelocity', 'xAcceleration', 'yAcceleration']]
Y = df ['ttc']

X_np = X.to_numpy()
Y_np = Y.to_numpy()

Y_pred = model.predict(X)

threshold = 2.0
# Wandle kontinuierliche TTC in Binärwerte um (1 = Gefahr, 0 = Sicher)
y_true_cls = (Y < threshold).astype(int)
y_pred_cls = (Y_pred < threshold).astype(int)

# --- 1. Klassifikations-Metriken (Gefahrenerkennung) ---
accuracy = accuracy_score(y_true_cls, y_pred_cls)
precision = precision_score(y_true_cls, y_pred_cls)
recall = recall_score(y_true_cls, y_pred_cls)
f1 = f1_score(y_true_cls, y_pred_cls)

# --- 2. Regressions-Metriken (Vorhersagegenauigkeit) ---
mse = mean_squared_error(Y_np, Y_pred)
rmse = np.sqrt(mse)
correlation = np.corrcoef(Y_np.flatten(), Y_pred.flatten())[0, 1]
std_dev = np.std(Y_np - Y_pred) # Standardabweichung der Fehler

print("--- Ergebnisse vom Random Forest ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f} (Vermeidung von Fehlalarmen)")
print(f"Recall: {recall:.4f} (Erkennungsrate echter Gefahren)")
print(f"F1-Score: {f1:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"Korrelation: {correlation:.4f}")
print(f"Standardabweichung (Fehler): {std_dev:.4f}")


importances = model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.plot(Y.values, label='Tatsächliche TTC (Ground Truth)', color='blue', linewidth=2)
plt.plot(Y_pred, label= 'Random forest vorhersage', color='orange', linestyle='--')
plt.axhline(y=2.0, color='red', label='Kritischer Schwellenwert (2.0s)', linestyle=':')
plt.title('Randomforst TTC Vorhersage')
plt.xlabel('Zeitverlauf (Frames)')
plt.ylabel('TTC')
plt.legend()
plt.grid(True, alpha=0.3)

plt.figure(figsize=(10, 6))
plt.title('Welche Faktoren beinflussen die Unfallwahrscheinlichkeit (TTC)?')
plt.bar(range(X.shape[1]), importances[indices], align='center', color='skyblue')
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45)
plt.ylabel('Wichtigkeit (Gini Importance)')
plt.tight_layout()

plt.show()