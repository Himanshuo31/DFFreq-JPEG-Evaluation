from pathlib import Path
import csv
import torch
from PIL import Image
from torchvision import transforms
from networks.resnet import resnet50

BASE = Path(r"C:\Users\LENOVO\Downloads\dalle")
MODEL_PATH = Path("checkpoints/model_epoch_last.pth")

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

def score(path):
    with Image.open(path) as img:
        x = transform(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        return model(x.to(device)).reshape(-1).item()

original_folder = BASE / "dalle" / "1_fake"
compressed_folder = BASE / "jpeg_experiments" / "quality_25" / "1_fake"

rows = []

for path in sorted(original_folder.glob("*")):
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        continue

    jpeg = compressed_folder / f"{path.stem}.jpg"
    if not jpeg.exists():
        continue

    before = score(path)
    after = score(jpeg)

    rows.append({
        "image": path.name,
        "original_score": before,
        "q25_score": after,
        "score_change": after - before,
        "original_pred": int(before > 0),
        "q25_pred": int(after > 0)
    })

if not rows:
    raise RuntimeError("No matching image pairs found.")

count = len(rows)
changes = [r["score_change"] for r in rows]
flips = sum(r["original_pred"] != r["q25_pred"] for r in rows)
fake_to_real = sum(
    r["original_pred"] == 1 and r["q25_pred"] == 0 for r in rows
)
real_to_fake = sum(
    r["original_pred"] == 0 and r["q25_pred"] == 1 for r in rows
)

with open("fake_q25_score_changes.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Matched image pairs: {count}")
print(f"Mean original score: {sum(r['original_score'] for r in rows) / count:.4f}")
print(f"Mean compressed score: {sum(r['q25_score'] for r in rows) / count:.4f}")
print(f"Mean score change: {sum(changes) / count:.4f}")
print(f"Scores decreased: {sum(x < 0 for x in changes)}")
print(f"Scores increased: {sum(x > 0 for x in changes)}")
print(f"Predictions changed: {flips}")
print(f"Fake to real: {fake_to_real}")
print(f"Real to fake: {real_to_fake}")
print("Detailed results saved to fake_q25_score_changes.csv")