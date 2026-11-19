import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("controlled_results.csv")

fake = df[df["Class"] == "1_fake"].sort_values(
    "JPEG_Quality", ascending=False
)

qualities = [100] + fake["JPEG_Quality"].tolist()
correct = [411] + fake["Compressed_Correct"].tolist()
accuracy = [x / 1000 * 100 for x in correct]

plt.figure(figsize=(8, 5))
plt.plot(qualities, accuracy, marker="o")
plt.xlabel("JPEG Quality")
plt.ylabel("AI Images Correctly Detected (%)")
plt.title("DFFreq Detection Performance Under JPEG Compression")
plt.xticks(qualities)
plt.ylim(0, 50)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("controlled_accuracy
Test-Path .\controlled_accuracy_chart.png
@'
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("controlled_results.csv")
fake = df[df["Class"] == "1_fake"].sort_values("JPEG_Quality", ascending=False)

qualities = [100] + fake["JPEG_Quality"].tolist()
correct = [411] + fake["Compressed_Correct"].tolist()
accuracy = [x / 1000 * 100 for x in correct]

plt.figure(figsize=(8, 5))
plt.plot(qualities, accuracy, marker="o")
plt.xlabel("JPEG Quality")
plt.ylabel("AI Images Correctly Detected (%)")
plt.title("DFFreq Detection Performance Under JPEG Compression")
plt.xticks(qualities)
plt.ylim(0, 50)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("controlled_accuracy_chart.png", dpi=300)
print("Graph saved successfully!")
