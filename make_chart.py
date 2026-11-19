import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("all_quality_results.csv")
df["Accuracy (%)"] = df["Accuracy"] * 100

plt.figure(figsize=(8, 5))
plt.bar(df["Quality"].astype(str), df["Accuracy (%)"])
plt.xlabel("Image condition")
plt.ylabel("Overall accuracy (%)")
plt.title("DFFreq Accuracy Under JPEG Compression")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig("accuracy_chart.png", dpi=200)
print("Saved accuracy_chart.png")
