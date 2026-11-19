import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("controlled_evaluation_results.csv")
summary = df.pivot(index="Quality", columns="Class", values="Accuracy_Percent")
summary["Overall"] = (summary["0_real"] + summary["1_fake"]) / 2
summary = summary.sort_index(ascending=False)

plt.figure(figsize=(9, 5))
plt.plot(summary.index, summary["0_real"], marker="o", label="Real Accuracy")
plt.plot(summary.index, summary["1_fake"], marker="o", label="Fake Accuracy")
plt.plot(summary.index, summary["Overall"], marker="o", label="Overall Accuracy")
plt.xlabel("JPEG Quality")
plt.ylabel("Accuracy (%)")
plt.title("DFFreq Performance under Controlled JPEG Compression")
plt.xticks([25, 50, 75, 95, 100])
plt.ylim(0, 105)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("controlled_jpeg_accuracy.png", dpi=300)
plt.close()
print("Chart saved successfully!")
