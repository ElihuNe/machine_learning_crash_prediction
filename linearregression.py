import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

data_path_folder = r'C:.\highD\data'

data_set_file = r'C:.\highD\data\35_tracks.csv'

df = pd.read_csv(data_set_file)

df = df[(df['ttc'] > 0) & (df['ttc'] <= 10)]

#df = df[df['dhw'] > 0]

X = df[['dhw']]
Y = df ['ttc']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

r_sq = r2_score(Y_test, Y_pred)

print("--- Ergebnisse der Linearen Regression ---")
print(f"R-Quadrat-Wert (R²): {r_sq:.4f}")
print(f"Achsenabschnitt (Intercept): {model.intercept_:.2f}")
print(f"Steigung (Coefficient für Länge): {model.coef_[0]:.2f}")

plt.figure(figsize=(10, 10))

plt.scatter(X_test, Y_test, color='blue', alpha=0.5, label='Real Datapoints')

plt.plot(X_test, Y_pred, color='red', linewidth=2, label=f'Regressionslinie ($y={model.intercept_:.2f} + {model.coef_[0]:.2f}x$)')

plt.title('Linear Regression on highD dataset')
plt.xlabel('vehicle velocity')
plt.ylabel('time to collision')
plt.legend()
plt.grid(True)
plt.show()