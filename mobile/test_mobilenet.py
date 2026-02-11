import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_large
from PIL import Image
import requests
from io import BytesIO

# -------------------------------------------------
# 1. Device
# -------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# -------------------------------------------------
# 2. Load pretrained MobileNetV3
# -------------------------------------------------
model = mobilenet_v3_large(weights="DEFAULT")
model.eval()
model.to(device)

# -------------------------------------------------
# 3. Image preprocessing
# -------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# -------------------------------------------------
# 4. Load image (change path if needed)
# -------------------------------------------------
# Option A: Local image
image = Image.open("mobile/cat.jpeg").convert("RGB")

# Option B: From URL
# url = "https://images.unsplash.com/photo-1518791841217-8f162f1e1131"
# image = Image.open(BytesIO(requests.get(url).content)).convert("RGB")

input_tensor = transform(image).unsqueeze(0).to(device)

# -------------------------------------------------
# 5. Inference
# -------------------------------------------------
with torch.no_grad():
    outputs = model(input_tensor)

# -------------------------------------------------
# 6. Top-5 Predictions
# -------------------------------------------------
probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
top5_prob, top5_catid = torch.topk(probabilities, 5)

# Load ImageNet labels
# labels = requests.get(
#    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
# ).text.split("\n")

with open("mobile/imagenet_classes.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

print("\nTop-5 Predictions:")
for i in range(5):
    print(f"{labels[top5_catid[i]]}: {top5_prob[i].item():.4f}")

# -------------------------------------------------
# 7. Extract Feature Embedding (important part)
# -------------------------------------------------
# Remove classification head
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
feature_extractor.to(device)

with torch.no_grad():
    features = feature_extractor(input_tensor)

features = features.view(features.size(0), -1)

print("\nFeature vector shape:", features.shape)