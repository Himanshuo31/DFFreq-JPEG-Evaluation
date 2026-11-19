from pathlib import Path
import csv
import torch
from PIL import Image
from torchvision import transforms
from networks.resnet import resnet50

DATA_ROOT = Path(r"C:\Users\LENOVO\Downloads\dalle\jpeg_experiments")
MODEL_PATH = Path("checkpoints/model_epoch_last.pth")
QUALITIES = [25]

device = torch.device("cpu")

# Use the original DFFreq architecture
model = resnet50(num_classes=1)
checkpoint = torch.load(
    MODEL_PATH, map_location=device, weights_only=False
)
model.load_state_dict(checkpoint, strict=True)
model.to(device)
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

    for class_name, actual_label in [("0_real", 0), ("1_fake", 1)]:
        folder = DATA_ROOT / f"quality_{quality}" / class_name
        files = [
            p for p in folder.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]

        for path in files:
            with Image.open(path) as img:
                tensor = transform(img.convert("RGB")).unsqueeze(0)

            with torch.no_grad():
                score = model(tensor.to(device)).reshape(-1).sigmoid().item()
   

            predicted = int(score > 0.5)
            total += 1
            correct += int(predicted == actual_label)

            if actual_label == 0:
                real_total += 1
                real_correct += int(predicted == 0)
            else:
                fake_total += 1
                fake_correct += int(predicted == 1)

    row = {
        "JPEG_Quality": quality,
        "Images": total,
        "Accuracy": correct / total,
        "Real_Accuracy": real_correct / real_total,
        "Fake_Accuracy": fake_correct / fake_total
    }
    results.append(row)

    print(
        f"Images evaluated: {total}"
        f"Quality {quality}: "
        f"Accuracy={row['Accuracy']:.4f}, "
        f"Real Accuracy={row['Real_Accuracy']:.4f}, "
        f"Fake Accuracy={row['Fake_Accuracy']:.4f}"
    )

with open("compression_results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\nResults saved to compression_results.csv")
