
from pathlib import Path
import csv
import torch
from PIL import Image
from torchvision import transforms
from networks.resnet import resnet50

BASE = Path(r"C:\Users\LENOVO\Downloads\dalle\controlled_jpeg")
QUALITIES = [100, 95, 75, 50, 25]
CLASSES = [("0_real", 0), ("1_fake", 1)]

MODEL_PATH = Path("checkpoints/model_epoch_last.pth")

device = torch.device("cpu")
model = resnet50(num_classes=1)
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device, weights_only=True)
)
model.eval()

transform = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

def predict(path):
    with Image.open(path) as img:
        x = transform(img.convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        score = model(x).reshape(-1).item()

    return int(score > 0)

results = []

for cls, label in CLASSES:
    baseline_folder = BASE / "quality_100" / cls

    for quality in QUALITIES[1:]:
        compressed_folder = BASE / f"quality_{quality}" / cls

        matched = 0
        baseline_correct = 0
        compressed_correct = 0
        predictions_changed = 0

        for original_path in sorted(baseline_folder.glob("*.jpg")):
            compressed_path = compressed_folder / original_path.name

            if not compressed_path.exists():
                continue

            baseline_prediction = predict(original_path)
            compressed_prediction = predict(compressed_path)

            matched += 1

            if baseline_prediction == label:
                baseline_correct += 1

            if compressed_prediction == label:
                compressed_correct += 1

            if baseline_prediction != compressed_prediction:
                predictions_changed += 1

        results.append({
            "Class": cls,
            "JPEG_Quality": quality,
            "Matched_Images": matched,
            "Q100_Correct": baseline_correct,
            "Compressed_Correct": compressed_correct,
            "Predictions_Changed": predictions_changed
        })

        print(
            f"{cls} | Q{quality} | Matched: {matched} | "
            f"Q100 correct: {baseline_correct} | "
            f"Compressed correct: {compressed_correct} | "
            f"Changed: {predictions_changed}"
        )

output = Path("controlled_paired_results.csv")

with output.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"\nSaved results to: {output}")
