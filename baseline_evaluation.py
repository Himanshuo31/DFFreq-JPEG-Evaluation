from pathlib import Path
import csv
import torch
from PIL import Image
from torchvision import transforms
from networks.resnet import resnet50

BASE = Path(r"C:\Users\LENOVO\Downloads\dalle\jpeg_experiments")
MODEL_PATH = Path("checkpoints/model_epoch_last.pth")
QUALITIES = [95, 75, 50, 25]

device = torch.device("cpu")
model = resnet50(num_classes=1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

transform = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict(path):
    with Image.open(path) as img:
        x = transform(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        return int(model(x).reshape(-1).item() > 0)

results = []

for cls, label in [("0_real", 0), ("1_fake", 1)]:
    baseline = BASE / "quality_100" / cls
    for q in QUALITIES:
        compressed = BASE / f"quality_{q}" / cls
        matched = baseline_correct = compressed_correct = changed = 0

        for p in baseline.glob("*.jpg"):
            cp = compressed / p.name
            if not cp.exists():
                continue
            a, b = predict(p), predict(cp)
            matched += 1
            baseline_correct += int(a == label)
            compressed_correct += int(b == label)
            changed += int(a != b)

        row = {
            "Class": cls, "JPEG_Quality": q, "Matched_Images": matched,
            "Baseline100_Correct": baseline_correct,
            "Compressed_Correct": compressed_correct,
            "Predictions_Changed": changed
        }
        results.append(row)
        print(row)

with open("controlled_results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\nSaved: controlled_results.csv")
