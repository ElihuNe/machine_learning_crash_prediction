import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error, confusion_matrix, ConfusionMatrixDisplay)
import numpy as np
import pandas as pd
import utils.save_split as save_split

model = joblib.load('data/logistic_regression/logistic_regression_model.pkl')

scaler = joblib.load('data/logistic_regression/scaler.pkl')

feature_names = ['dhw', 'xVelocity', 'yVelocity', 'width', 'height', 'precedingXVelocity', 'xAcceleration', 'yAcceleration']

X, Y = save_split.load_test_data('data/logistic_regression')

Y_pred = model.predict(X)
Y_prob = model.predict_proba(X)[:,1]
# --- 1. Klassifikations-Metriken (Gefahrenerkennung) ---
accuracy = accuracy_score(Y, Y_pred)
precision = precision_score(Y, Y_pred)
recall = recall_score(Y, Y_pred)
f1 = f1_score(Y, Y_pred)

mse = mean_squared_error(Y, Y_pred)
rmse = np.sqrt(mse)
correlation = np.corrcoef(Y.flatten(), Y_pred.flatten())[0, 1]
std_dev = np.std(Y - Y_pred) # Standardabweichung der Fehler

print("\n--- Ergebnisse der Logistischen Regression ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f} (Vermeidung von Fehlalarmen)")
print(f"Recall: {recall:.4f} (Erkennungsrate echter Gefahren)")
print(f"F1-Score: {f1:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"Korrelation: {correlation:.4f}")
print(f"Standardabweichung (Fehler): {std_dev:.4f}")

metrics_vals = [accuracy, precision, recall, f1, mse, rmse, correlation, std_dev]
metrics_names = [f"Accuracy\n{accuracy:.4f}", f"Precision\n{precision:.4f}", f"Recall\n{recall:.4f}", f"F1-Score\n{f1:.4f}", f"MSE\n{mse:.4f}", f"RMSE\n{rmse:.4f}", f"Korrelation\n{correlation:.4f}", f"Standardabweichung\n{std_dev:.4f}"]

importances = model.coef_[0]
indices = np.argsort(importances)[::-1]
limit = 1000

# Vergleichs plot
plt.figure(figsize=(12, 6))
plt.plot(Y, label='Tatsächliche Kritikalität (Ground Truth)', color='blue', linewidth=2)
plt.plot(Y_pred, label= 'Logistische Regression Vorhersage', color='orange', linestyle='--')
plt.axhline(y=0.5, color='red', label='Kritischer Schwellenwert (0.5)', linestyle=':')
plt.title('Logistische Regression Kritikalitätsvorhersage')
plt.xlabel('Probenindex')
plt.ylabel('Kritikalität (1 = Kritisch, 0 = Sicher)')
plt.legend()
plt.xlim(0, limit)

# Feature importance
plt.figure(figsize=(10, 6))
plt.title('Feature-Wichtigkeit der Logistischen Regression')
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
plt.xlabel('Feature')
plt.ylabel('Koeffizient')
plt.grid()

# Statistische Metiken Balken diagramm
plt.figure(figsize=(12, 6))
plt.bar(metrics_names, metrics_vals)
plt.title("Ergebnisse der Logistischen Regression")
plt.xlabel("Metriken")
plt.grid(True, alpha=0.5)

# Confusion Matrix
con = confusion_matrix(Y, Y_pred)
cm_plot = ConfusionMatrixDisplay(con, display_labels=[0,1])
cm_plot.plot()


plt.show()