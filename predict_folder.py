import os
import sys
import torch
from PIL import Image
from torchvision import transforms
from networks.resnet import resnet50

MODEL_PATH = "./checkpoints/model_epoch_last.pth"
EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

transform = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

model = resnet50(num_classes=1)
model.load_state_dict(
    torch.load(MODEL_PATH, map_location="cpu"),
    strict=True
)
model.eval()

folder = sys.argv[1] if len(sys.argv) > 1 else "./my_test"

files = []
for root, _, names in os.walk(folder):
    for name in names:
        if name.lower().endswith(EXTENSIONS):
            files.append(os.path.join(root, name))

if not files:
    print("No images found.")
    sys.exit(0)

print(f"\nFound {len(files)} images\n")

for path in sorted(files):
    try:
        image = Image.open(path).convert("RGB")
        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            logit = model(tensor).item()
            score = torch.sigmoid(torch.tensor(logit)).item()

        prediction = "AI-generated" if score >= 0.5 else "Real"

        print(f"Image: {path}")
        print(f"Raw logit: {logit:.6f}")
        print(f"AI score: {score:.6f}")
        print(f"Prediction: {prediction}\n")

    except Exception as e:
        print(f"Error processing {path}: {e}\n")