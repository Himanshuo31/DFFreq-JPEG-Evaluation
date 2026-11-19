import pandas as pd

original = pd.read_csv("paired_results.csv")
controlled = pd.read_csv("controlled_results.csv")

original["Experiment"] = "Original vs compressed"
controlled["Experiment"] = "JPEG100 baseline vs compressed"

combined = pd.concat([original, controlled], ignore_index=True)
combined.to_csv("final_combined_results.csv", index=False)

print("Combined results saved!")
print(combined.to_string(index=False))
