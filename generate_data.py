"""Generate synthetic sales data for demand forecasting.

Extends the original sales_data.csv from sales-data-analysis
with a yearly growth trend (20% over the year) to give the
forecasting models a real trend component to learn.
"""
import pandas as pd
import numpy as np

np.random.seed(42)  # reproducibility — same data every run

dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
products = ["Sensor A", "Sensor B", "Controller X", "Cable Set"]

rows = []
for date in dates:
    for product in products:
        base = {"Sensor A": 120, "Sensor B": 80,
                "Controller X": 45, "Cable Set": 200}[product]

        # Three signal components:
        trend = 1 + 0.20 * (date.dayofyear / 365)    # +20% growth over the year
        season = 1 + 0.30 * np.sin(2 * np.pi * date.dayofyear / 365)
        noise = np.random.normal(1.0, 0.15)

        units = max(0, int(base * trend * season * noise))
        rows.append({"date": date, "product": product, "units_sold": units})

df = pd.DataFrame(rows)
df.to_csv("sales_data.csv", index=False)

print(f"Created sales_data.csv with {len(df)} rows")
print(df.head())
print()
print(f"Sensor A total: Jan {df[(df['product']=='Sensor A') & (df['date'].dt.month==1)]['units_sold'].sum()}, "
      f"Dec {df[(df['product']=='Sensor A') & (df['date'].dt.month==12)]['units_sold'].sum()}")