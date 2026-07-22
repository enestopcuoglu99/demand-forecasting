"""Visualize the three components of the time series:
Trend + Seasonality + Noise = Observed data.

This is the foundation of every forecasting method:
if you can see the components, you can model them.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("sales_data.csv", parse_dates=["date"])

# Focus on one product to keep the story clean
product = "Sensor A"
data = df[df["product"] == product].set_index("date")["units_sold"]

# --- Reconstruct the "true" components we baked in ---
base = 120
day_of_year = data.index.dayofyear.values

true_trend = base * (1 + 0.20 * day_of_year / 365)
true_season = base * (1 + 0.30 * np.sin(2 * np.pi * day_of_year / 365)) - base
# noise = observed - (trend + seasonality component around base)
combined = true_trend + true_season
noise = data.values - combined

# --- Plot: 4 stacked panels ---
fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)

axes[0].plot(data.index, data.values, color="steelblue", linewidth=0.7)
axes[0].set_title(f"1) Observed data — {product} (raw signal)")
axes[0].set_ylabel("Units sold")
axes[0].grid(alpha=0.3)

axes[1].plot(data.index, true_trend, color="darkred", linewidth=2)
axes[1].set_title("2) Trend component (+20% growth over the year)")
axes[1].set_ylabel("Units")
axes[1].grid(alpha=0.3)

axes[2].plot(data.index, true_season, color="darkgreen", linewidth=1.5)
axes[2].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
axes[2].set_title("3) Seasonal component (±30% sine wave)")
axes[2].set_ylabel("Deviation")
axes[2].grid(alpha=0.3)

axes[3].plot(data.index, noise, color="gray", linewidth=0.5)
axes[3].axhline(y=0, color="black", linestyle="--", alpha=0.5)
axes[3].set_title("4) Noise component (random ±15%)")
axes[3].set_ylabel("Residual")
axes[3].set_xlabel("Date")
axes[3].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("decomposition.png", dpi=150)
plt.show()