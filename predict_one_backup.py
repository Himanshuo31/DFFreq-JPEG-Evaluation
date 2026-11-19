import sys
import torch
from PIL import Image
from torchvision import transforms
from networks.resnet import resnet50

MODEL_PATH = "./checkpoints/model_epoch_last.pth"

transform = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

model = resnet50(num_classes=1)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"), strict=True)
model.eval()

image_path = sys.argv[1] if len(sys.argv) > 1 else "./my_test/1_fake/ai_generated_car.jpg"

image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    logit = model(input_tensor).item(); score = torch.sigmoid(torch.tensor(logit)).item()

print(f"Image: {image_path}")
print(f"Raw logit: {logit:.6f}"); print(f"Model score: {score:.10f}")
print(f"Prediction: {'AI-generated' if score >= 0.5 else 'Real'}")


