import matplotlib.pyplot as plt
import pandas as pd
import torch
import joblib
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
    sample_inputs = X_test_t[test_range]
    predictions = model(sample_inputs)
    actuals = Y_test[test_range]

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