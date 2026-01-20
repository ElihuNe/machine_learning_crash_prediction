import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import glob
import joblib
import utils.save_split as save_split

class TTC_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(TTC_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
    
def lstm_data(df, window_size=50, scaler=StandardScaler()):
    features = ['dhw', 'xVelocity', 'yVelocity', 'width', 'height', 'precedingXVelocity', 'xAcceleration', 'yAcceleration']
    target = 'ttc'

    df[features] = scaler.fit_transform(df[features])

    X_list, y_list = [], []

    grouped = df.groupby('id')

    for _, group in grouped:

        if len(group) > window_size:
            feature_data = group[features].values
            target_data = group[target].values

            for i in range(len(group) - window_size):
                X_list.append(feature_data[i:i + window_size])
                y_list.append(target_data[i + window_size])

    return np.array(X_list), np.array(y_list)


if __name__ == "__main__":

    data_path_folder = r'C:.\highD\data'
    search_pattern = os.path.join(data_path_folder, '*_tracks.csv')
    all_files = glob.glob(search_pattern)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_dfs = []

    for file_path in all_files:
        base_name = os.path.basename(file_path)
        recording_id = base_name.split('_')[0]

        df_tmp = pd.read_csv(file_path, usecols=['id', 'ttc', 'dhw', 'xVelocity', 'yVelocity', 'width', 'height', 'precedingXVelocity', 'xAcceleration', 'yAcceleration'])
        df_tmp['id'] = recording_id + '_' + df_tmp['id'].astype(str)

        df_tmp = df_tmp[(df_tmp['ttc'] > 0) & (df_tmp['ttc'] <= 10)]

        all_dfs.append(df_tmp)

    df = pd.concat(all_dfs, ignore_index=True)

    #EMA glättung von bestimmten features
    #df['xAcceleration'] = df.groupby('id')['xAcceleration'].transform(lambda x: x.ewm(span=5, adjust=False).mean())
    #df['yAcceleration'] = df.groupby('id')['yAcceleration'].transform(lambda x: x.ewm(span=5, adjust=False).mean())

    scaler = StandardScaler()

    X_final, y_final = lstm_data(df, scaler=scaler)

    X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42)

    save_split.save_processed_split(X_train, X_test, y_train, y_test, scaler, 'data/lstm')

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    model = TTC_LSTM(input_size=8, hidden_size=128, num_layers=2)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 20
    batch_size = 64

    train_data = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(train_loader):.4f}')

    model_path = 'data/lstm/ttc_lstm_model.pth'
    torch.save(model.state_dict(), model_path)