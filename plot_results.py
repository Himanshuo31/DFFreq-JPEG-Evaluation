import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("paired_results.csv")

df["Accuracy"] = (
    df["Compressed_Correct"] / df["Matched_Images"] * 100
)

labels = ["Original", "Q95", "Q75", "Q50", "Q25"]

real = [99.4] + df[df["Class"] == "0_real"]["Accuracy"].tolist()
fake = [69.4] + df[df["Class"] == "1_fake"]["Accuracy"].tolist()

plt.figure(figsize=(9, 5))

x = range(len(labels))
width = 0.35

plt.bar(
    [i - width / 2 for i in x],
    real,
    width,
    label="Real Images"
)

plt.bar(
    [i + width / 2 for i in x],
    fake,
    width,
    label="AI-Generated Images"
)

plt.xlabel("JPEG Quality")
plt.ylabel("Detection Accuracy (%)")
plt.title("Effect of JPEG Compression on DFFreq Detection")
plt.xticks(list(x), labels)
plt.ylim(0, 105)
plt.legend()
plt.tight_layout()

plt.savefig("jpeg_compression_results.png", dpi=300)
plt.show()

print("Graph saved as jpeg_compression_results.png")