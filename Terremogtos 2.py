import numpy as np
import pandas as pd
df = pd.read_csv("PRVI_SARotD50[1].csv")
df = df[df["Network"] == "PR"]
eqs = (
    df[["EarthquakeId", "EarthquakeTime", "EarthquakeMagnitude"]]
    .drop_duplicates()
    <copy()
)

eqs["EarthquakeTime"] = pd.to_datetime(eqs["EarthquakeTime"])
eqs = eqs.sort_values("EarthquakeTime").reset_index(drop=True)
print(eqs.head())

eqs["NextTime"] = eqs["EarthquakeTime"].shift(-1)
eqs["DeltaT_days"] = (eqs["NextTime"] - eqs["EarthquakeTime"]).dt.total_seconds() / (
    24 * 3600
)
eqs["Mp"] = eqs["EarthquakeMagnitude"]
eqs_clean = eqs.dropna(subset=["DeltaT_days", "Mp"]).copy()
eqs_clean["logT"] = np.log10(eqs_clean["DeltaT_days"])
print(eqs_clean[["EarthquakeId", "EarthquakeTime", "Mp", "DeltaT_days", "logT"]].head())

from scipy.stats import linregress
x = eqs_clean["Mp"]
y = eqs_clean["logT"]

slope, intercept, r_value, p_value, std_err = linregress(x, y)

c = slope
a = intercept

print(f"Empirical constant c (slope): {c:.4f}")
print(f"Empirical constant a (intercept): {a:.4f}")
print(f"Correlation coefficient (r): {r_value:.4f}")
print(f"P-value: {p_value:.4e}")
print(f"R-squared: {r_value**2:.4f}")
print(f"\nCompleted Formula: logT = {c:.4f} * Mp + {a:.4f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.scatter(
    x, y, color="royalblue", alpha=0.7, edgecolor="k", label="Observed Events"
)
x_vals = np.linspace(x.min(), x.max(), 100)
y_vals = c * x_vals + a
plt.plot(
    x_vals,
    y_vals,
    color="crimson",
    linewidth=2,
    label=f"Fit: logT = {c:.2f}Mp + {a:.2f}",
)

plt.xlabel("Preceding Magnitude ($M_p$)", fontsize=12)
plt.ylabel(r"$\log_{10}(T)$ [Days]", fontsize=12)
plt.title("Time-Predictable Recurrence Model Fit", fontsize=14)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
