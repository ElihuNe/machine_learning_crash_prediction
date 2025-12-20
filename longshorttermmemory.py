import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data_path_folder = r'C:.\highD\data'

data_set_file = r'C:.\highD\data\35_tracks.csv'

df = pd.read_csv(data_set_file)

df = df[(df['ttc'] > 0) & (df['ttc'] <= 10)]

features = ['xVelocity', 'dhw']
target = 'ttc'

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])
labels = df[target].values

def create_sequences(data, labels, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(labels[i + window_size])
    return np.array(X), np.array(y)

window_size = 25

X_seq, y_seq = create_sequences(df_scaled, labels, window_size)

X_train_t = torch.tensor(X_seq, dtype=torch.float32)
y_train_t = torch.tensor(y_seq, dtype=torch.float32).view(-1, 1)

class TTC_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(TTC_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = TTC_LSTM(input_size=2, hidden_size=64, num_layers=2)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
outputs = model(X_train_t[:1000])
loss = criterion(outputs, y_train_t[:1000])
optimizer.zero_grad()
loss.backward()
optimizer.step()

model.eval()
with torch.no_grad():

    test_range = range(500, 800)
    sample_inputs = X_train_t[test_range]
    predictions = model(sample_inputs).numpy()
    actuals = y_train_t[test_range].numpy()

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