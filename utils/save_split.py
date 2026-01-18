import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split

def save_processed_split(X_train, X_test, y_train, y_test, scaler, folder='data/processed'):
    """
    Speichert den Train-Test-Split und den Scaler in einem dedizierten Ordner.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    # 1. Speichern der NumPy Arrays (komprimiert)
    # Wir speichern alles in einer .npz Datei
    np.savez_compressed(
        os.path.join(folder, 'split_data.npz'), 
        X_train=X_train, 
        X_test=X_test, 
        y_train=y_train, 
        y_test=y_test
    )
    
    # 2. Speichern des Scalers (wichtig für die Inferenz im Eval-Modul)
    if scaler is not None:
        joblib.dump(scaler, os.path.join(folder, 'scaler.pkl'))
    
    print(f"Erfolg: Split-Daten und Scaler wurden in '{folder}' gespeichert.")

def load_test_data(folder='data/processed'):
    """
    Lädt nur die Testdaten für das Evaluations-Modul.
    """
    data_path = os.path.join(folder, 'split_data.npz')
    if not os.path.exists(data_path):
        raise FileNotFoundError("Keine Split-Daten gefunden! Erst Training/Split ausführen.")
    
    data = np.load(data_path)
    return data['X_test'], data['y_test']
