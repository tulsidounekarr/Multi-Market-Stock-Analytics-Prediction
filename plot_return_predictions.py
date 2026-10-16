
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "outputs/return_predictions.csv",
    parse_dates=["Date"]
)

plt.figure(figsize=(13, 6))
plt.plot(
    df["Date"], df["Actual_Close"],
    label="Actual Price", linewidth=2
)
plt.plot(
    df["Date"], df["Predicted_Close"],
    label="Return Model", linewidth=1.5
)
plt.plot(
    df["Date"], df["Baseline_Close"],
    label="Baseline (Today's Close)", alpha=0.8
)

plt.title("TCS: Return Model vs Actual Price vs Baseline")
plt.xlabel("Date")
plt.ylabel("Price (INR)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/return_model_comparison.png", dpi=150)
plt.show()

print("Chart saved to outputs/return_model_comparison.png")
