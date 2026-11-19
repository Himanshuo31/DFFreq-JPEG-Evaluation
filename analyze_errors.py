
import re
import csv

input_file = "evaluation_output.txt"
output_file = "misclassified_fake.csv"

pattern = re.compile(
    r"(.+\.png|.+\.jpg|.+\.jpeg) \| Actual: Fake "
    r"\| Score: ([0-9.]+) \| Predicted: Real",
    re.IGNORECASE
)

rows = []

with open(input_file, "r", encoding="utf-16") as f:
    for line in f:
        match = pattern.search(line.strip())
        if match:
            rows.append([match.group(1), float(match.group(2))])

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Filename", "AI_Score", "Actual", "Predicted"])
    for filename, score in rows:
        writer.writerow([filename, score, "Fake", "Real"])

print("Misclassified fake images:", len(rows))
print("CSV saved to:", output_file)
