from pathlib import Path
import csv
import torch
from PIL import Image
from torchvision import transforms
from networks.resnet import resnet50

BASE = Path(r"C:\Users\LENOVO\Downloads\dalle")
MODEL_PATH = Path("checkpoints/model_epoch_last.pth")
QUALITIES = [95, 75, 50, 25]

device = torch.device("cpu")
model = resnet50(num_classes=1)
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device, weights_only=False),
    strict=True
)
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

def get_raw_score(path):
    with Image.open(path) as img:
        tensor = transform(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        return model(tensor.to(device)).reshape(-1).item()

results = []

for class_name, label in [("0_real", 0), ("1_fake", 1)]:
    original_folder = BASE / "dalle" / class_name
    original_files = {
        p.stem: p for p in original_folder.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }

    for quality in QUALITIES:
        compressed_folder = (
            BASE / "jpeg_experiments" / f"quality_{quality}" / class_name
        )
        matched = 0
        correct_original = 0
        correct_compressed = 0
        changed = 0

        for original_stem, original_path in original_files.items():
            compressed_path = compressed_folder / f"{original_stem}.jpg"
            if not compressed_path.is_file():
                continue

            original_pred = int(get_raw_score(original_path) > 0)
            compressed_pred = int(get_raw_score(compressed_path) > 0)

            matched += 1
            correct_original += int(original_pred == label)
            correct_compressed += int(compressed_pred == label)
            changed += int(original_pred != compressed_pred)

        results.append({
            "Class": class_name,
            "JPEG_Quality": quality,
            "Matched_Images": matched,
            "Original_Correct": correct_original,
            "Compressed_Correct": correct_compressed,
            "Predictions_Changed": changed
        })

        print(
            f"{class_name}, Q{quality}: matched={matched}, "
            f"original_correct={correct_original}, "
            f"compressed_correct={correct_compressed}, "
            f"predictions_changed={changed}"
        )

with open("paired_results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\nResults saved to paired_results.csv")