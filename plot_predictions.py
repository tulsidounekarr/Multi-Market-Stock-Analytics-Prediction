
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("outputs/predictions.csv", parse_dates=["Date"])

plt.figure(figsize=(14, 6))
plt.plot(
    df["Date"], df["Actual_Close"],
    label="Actual Closing Price", linewidth=2
)
plt.plot(
    df["Date"], df["Predicted_Close"],
    label="Random Forest Prediction", linewidth=1.5
)

plt.title("TCS Stock: Actual vs Predicted Closing Price")
plt.xlabel("Date")
plt.ylabel("Price (INR)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/actual_vs_predicted.png", dpi=150)
plt.show()

print("Chart saved to outputs/actual_vs_predicted.png")
