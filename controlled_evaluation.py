
from pathlib import Path
import csv
import torch
from PIL import Image
from torchvision import transforms
from sklearn.metrics import accuracy_score
from networks.resnet import resnet50

BASE = Path(r"C:\Users\LENOVO\Downloads\dalle\controlled_jpeg")
MODEL_PATH = Path("checkpoints/model_epoch_last.pth")
QUALITIES = [100, 95, 75, 50, 25]
CLASSES = [("0_real", 0), ("1_fake", 1)]

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

for quality in QUALITIES:
    for cls, label in CLASSES:
        folder = BASE / f"quality_{quality}" / cls

        files = sorted(folder.glob("*.jpg"))
        correct = 0

        for path in files:
            prediction = predict(path)
            if prediction == label:
                correct += 1

        accuracy = correct / len(files) if files else 0

        results.append({
            "Quality": quality,
            "Class": cls,
            "Total_Images": len(files),
            "Correct": correct,
            "Accuracy_Percent": round(accuracy * 100, 2)
        })

        print(
            f"Q{quality} | {cls} | "
            f"Correct: {correct}/{len(files)} | "
            f"Accuracy: {accuracy * 100:.2f}%"
        )

output = Path("controlled_evaluation_results.csv")

with output.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"\nSaved results to: {output}")
