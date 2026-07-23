"""SARIMA forecast — ARIMA extended with seasonality.

The basic ARIMA missed the seasonal pattern. SARIMA explicitly
models both the trend and the yearly cycle.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("sales_data.csv", parse_dates=["date"])
data = df[df["product"] == "Sensor A"].set_index("date")["units_sold"]

train = data.iloc[:300]
test = data.iloc[300:]
horizon = len(test)


def mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100


# --- Fit SARIMA ---
# order = (p, d, q) for the non-seasonal part
# seasonal_order = (P, D, Q, s) for the seasonal part; s = season length in days
print("Fitting SARIMA model (may take 30-60 seconds)...")
model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 365))
fitted = model.fit(disp=False)

forecast = fitted.forecast(steps=horizon)
forecast.index = test.index

error = mape(test.values, forecast.values)
print(f"SARIMA(1,1,1)(1,1,1,365) -> MAPE = {error:.2f}%")


# --- Plot ---
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(train.index, train.values, color="steelblue",
        linewidth=0.8, label="Training data")
ax.plot(test.index, test.values, color="black",
        linewidth=1.2, label="Actual (test)")
ax.plot(forecast.index, forecast.values, color="darkred",
        linewidth=2, label=f"SARIMA  MAPE={error:.1f}%")

ax.set_title("SARIMA Forecast — Sensor A")
ax.set_ylabel("Units sold")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("sarima_forecast.png", dpi=150)
plt.show()