import matplotlib.pyplot as plt
import pandas as pd
import torch
import joblib
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error, confusion_matrix, ConfusionMatrixDisplay)
import numpy as np
from longshorttermmemory import TTC_LSTM, lstm_data
import utils.save_split as save_split

model = TTC_LSTM(input_size=8, hidden_size=128, num_layers=2)

model.load_state_dict(torch.load('data/lstm/ttc_lstm_model.pth'))

scaler = joblib.load('data/lstm/scaler.pkl')

X_test, Y_test = save_split.load_test_data('data/lstm')
X_test_t = torch.tensor(X_test, dtype=torch.float32)


model.eval()
with torch.no_grad():

    test_range = range(500, 800)
    sample_inputs = X_test_t
    predictions = model(sample_inputs)
    actuals = Y_test

np_predictions = np.array(actuals).flatten()
np_actuals = np.array(predictions).flatten()

threshold = 2.0
# Wandle kontinuierliche TTC in Binärwerte um (1 = Gefahr, 0 = Sicher)
y_true_cls = (np_predictions < threshold).astype(int)
y_pred_cls = (np_actuals < threshold).astype(int)

# --- 1. Klassifikations-Metriken (Gefahrenerkennung) ---
accuracy = accuracy_score(y_true_cls, y_pred_cls)
precision = precision_score(y_true_cls, y_pred_cls)
recall = recall_score(y_true_cls, y_pred_cls)
f1 = f1_score(y_true_cls, y_pred_cls)

# --- 2. Regressions-Metriken (Vorhersagegenauigkeit) ---
mse = mean_squared_error(np_actuals, np_predictions)
rmse = np.sqrt(mse)
correlation = np.corrcoef(np_actuals.flatten(), np_predictions.flatten())[0, 1]
std_dev = np.std(np_actuals - np_predictions)

print("--- Ergebnisse vom Long-Short-Term-Memory ---")
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

limit = 1000

# Vergleichs plot
plt.figure(figsize=(12, 6))
plt.plot(actuals, label='Tatsächliche TTC (Ground Truth)', color='blue', linewidth=2)
plt.plot(predictions, label='LSTM Vorhersage', color='orange', linestyle='--')
plt.axhline(y=2.0, color='red', linestyle=':', label='Kritischer Schwellenwert (2.0s)')
plt.title('LSTM Zeitreihen_Vorhersage: TTC über Frames')
plt.xlabel('Zeitverlauf (Frames)')
plt.ylabel('Time-to-Collision (s)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, limit)

# Statistische Metiken Balken diagramm
plt.figure(figsize=(12, 6))
plt.bar(metrics_names, metrics_vals)
plt.title("Ergebnisse vom Long-Short-Term-Memory")
plt.xlabel("Metriken")
plt.grid(True, alpha=0.5)

# Confusion Matrix
con = confusion_matrix(y_true_cls, y_pred_cls)
cm_plot = ConfusionMatrixDisplay(con, display_labels=[0,1])
cm_plot.plot()

plt.show()