
import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from sklearn.metrics import accuracy_score, average_precision_score
from networks.resnet import resnet50

MODEL_PATH = "./checkpoints/model_epoch_last.pth"
DATA_DIR = r"C:\Users\LENOVO\Downloads\dalle\dalle"

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

y_true, y_pred = [], []

for class_name, label in [("0_real", 0), ("1_fake", 1)]:
    class_dir = os.path.join(DATA_DIR, class_name)

    for filename in os.listdir(class_dir):
        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp", ".bmp")
        ):
            path = os.path.join(class_dir, filename)

            try:
                image = Image.open(path).convert("RGB")
                tensor = transform(image).unsqueeze(0)

                with torch.no_grad():
                    score = model(tensor).sigmoid().item()

                y_true.append(label)
                y_pred.append(score)

                actual = "Fake" if label == 1 else "Real"
                predicted = "Fake" if score > 0.5 else "Real"

                print(
                    f"{filename} | Actual: {actual} | "
                    f"Score: {score:.6f} | Predicted: {predicted}"
                )

            except Exception as e:
                print(f"Error: {path} | {e}")

if len(set(y_true)) < 2:
    print("\nBoth real and fake images are required.")
else:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    predicted = (y_pred > 0.5).astype(int)

    print("\n--- Evaluation Results ---")
    print(f"Images evaluated: {len(y_true)}")
    print(f"Accuracy: {accuracy_score(y_true, predicted):.4f}")
    print(
        "Real Accuracy: "
        f"{accuracy_score(y_true[y_true == 0], predicted[y_true == 0]):.4f}"
    )
    print(
        "Fake Accuracy: "
        f"{accuracy_score(y_true[y_true == 1], predicted[y_true == 1]):.4f}"
    )
    print(f"Average Precision (AP): {average_precision_score(y_true, y_pred):.4f}")
