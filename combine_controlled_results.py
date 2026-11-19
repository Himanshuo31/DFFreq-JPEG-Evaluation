
import pandas as pd

accuracy = pd.read_csv("controlled_evaluation_results.csv")
paired = pd.read_csv("controlled_paired_results.csv")

combined = paired.merge(
    accuracy,
    left_on=["Class", "JPEG_Quality"],
    right_on=["Class", "Quality"],
    how="left"
)

combined = combined.drop(columns=["Quality"])

combined.to_csv("final_controlled_results.csv", index=False)

print("\nFinal Controlled Results:\n")
print(combined.to_string(index=False))

print("\nSaved: final_controlled_results.csv")
