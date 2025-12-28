import matplotlib.pyplot as plt
import pandas as pd
import torch
import joblib
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error)
import numpy as np
from longshorttermmemory import TTC_LSTM, lstm_data

model = TTC_LSTM(input_size=8, hidden_size=64, num_layers=2)

model.load_state_dict(torch.load('ttc_lstm_model.pth'))

data_path = r'C:.\highD\data\04_tracks.csv'
df = pd.read_csv(data_path)

df = df[(df['ttc'] > 0) & (df['ttc'] <= 10)]

scaler = joblib.load('lstm_scaler.pkl')

X_test, Y_test = lstm_data(df, window_size=25, scaler=scaler)
X_test_t = torch.tensor(X_test, dtype=torch.float32)


model.eval()
with torch.no_grad():

    test_range = range(500, 800)
    sample_inputs = X_test_t#[test_range]
    predictions = model(sample_inputs)
    actuals = Y_test#[test_range]

np_predictions = np.array(actuals)
np_actuals = np.array(predictions)

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
std_dev = np.std(np_actuals - np_predictions) # Standardabweichung der Fehler

print("--- Ergebnisse vom Long-Short-Term-Memory ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f} (Vermeidung von Fehlalarmen)")
print(f"Recall: {recall:.4f} (Erkennungsrate echter Gefahren)")
print(f"F1-Score: {f1:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"Korrelation: {correlation:.4f}")
print(f"Standardabweichung (Fehler): {std_dev:.4f}")

plt.figure(figsize=(12, 6))

plt.plot(actuals, label='Tatsächliche TTC (Ground Truth)', color='blue', linewidth=2)
plt.plot(predictions, label='LSTM Vorhersage', color='orange', linestyle='--')
plt.axhline(y=2.0, color='red', linestyle=':', label='Kritischer Schwellenwert (2.0s)')
plt.title('LSTM Zeitreihen_Vorhersage: TTC über Frames')
plt.xlabel('Zeitverlauf (Frames)')
plt.ylabel('Time-to-Collision (s)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()