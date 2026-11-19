from pathlib import Path
import csv
import torch
from PIL import Image
from torchvision import transforms
from networks.resnet import resnet50

BASE = Path(r"C:\Users\LENOVO\Downloads\dalle")
MODEL_PATH = Path("checkpoints/model_epoch_last.pth")
QUALITIES = ["original", 95, 75, 50, 25]

device = torch.device("cpu")
model = resnet50(num_classes=1)
checkpoint = torch.load(
    MODEL_PATH, map_location=device, weights_only=False
)
model.load_state_dict(checkpoint, strict=True)
model.eval()

transform = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

results = []

for quality in QUALITIES:
    total = correct = 0
    real_total = fake_total = 0
    real_correct = fake_correct = 0

    for class_name, label in [("0_real", 0), ("1_fake", 1)]:
        if quality == "original":
            folder = BASE / "dalle" / class_name
        else:
            folder = BASE / "jpeg_experiments" / f"quality_{quality}" / class_name

        files = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

        for path in files:
            with Image.open(path) as img:
                tensor = transform(img.convert("RGB")).unsqueeze(0)

            with torch.no_grad():
                raw = model(tensor.to(device)).reshape(-1).item()

            predicted = int(raw > 0)
            total += 1
            correct += int(predicted == label)

            if label == 0:
                real_total += 1
                real_correct += int(predicted == 0)
            else:
                fake_total += 1
                fake_correct += int(predicted == 1)

    row = {
        "Quality": quality,
        "Images": total,
        "Accuracy": correct / total,
        "Real_Accuracy": real_correct / real_total,
        "Fake_Accuracy": fake_correct / fake_total
    }
    results.append(row)

    print(
        f"Quality {quality}: Images={total}, "
        f"Accuracy={row['Accuracy']:.4f}, "
        f"Real={row['Real_Accuracy']:.4f}, "
        f"Fake={row['Fake_Accuracy']:.4f}"
    )

with open("all_quality_results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\nResults saved to all_quality_results.csv")