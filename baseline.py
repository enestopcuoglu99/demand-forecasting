"""Moving average baseline forecast.

The 'honest but naive' benchmark: predict tomorrow as the
average of the last N days. Every smarter model must beat this.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("sales_data.csv", parse_dates=["date"])
data = df[df["product"] == "Sensor A"].set_index("date")["units_sold"]

# --- Train/test split ---
# Learn from first 300 days, forecast the last 65
train = data.iloc[:300]
test = data.iloc[300:]


def moving_average_forecast(train_data, window, horizon):
    """Forecast `horizon` days ahead using the last `window` days average.

    Simple approach: repeat the same average value for all future days.
    """
    last_avg = train_data.iloc[-window:].mean()
    return pd.Series(
        [last_avg] * horizon,
        index=pd.date_range(train_data.index[-1] + pd.Timedelta(days=1),
                            periods=horizon, freq="D")
    )


def mape(actual, predicted):
    """Mean Absolute Percentage Error — the forecasting standard."""
    return np.mean(np.abs((actual - predicted) / actual)) * 100


# --- Try three window sizes ---
horizon = len(test)
results = {}
for N in [7, 30, 90]:
    forecast = moving_average_forecast(train, window=N, horizon=horizon)
    error = mape(test.values, forecast.values)
    results[N] = (forecast, error)
    print(f"N={N:3d} days -> MAPE = {error:.2f}%")

# --- Plot ---
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(train.index, train.values, color="steelblue",
        linewidth=0.8, label="Training data")
ax.plot(test.index, test.values, color="black",
        linewidth=1.2, label="Actual (test)")

colors = {7: "darkred", 30: "darkorange", 90: "darkgreen"}
for N, (fc, err) in results.items():
    ax.plot(fc.index, fc.values, color=colors[N], linewidth=2,
            linestyle="--", label=f"MA(N={N})  MAPE={err:.1f}%")

ax.set_title("Moving Average Forecasts — Sensor A")
ax.set_ylabel("Units sold")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("baseline_forecast.png", dpi=150)
plt.show()