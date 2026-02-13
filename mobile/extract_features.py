import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_large
from PIL import Image

# --------------------------------------------
# Device
# --------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# --------------------------------------------
# Load MobileNet (pretrained)
# --------------------------------------------
model = mobilenet_v3_large(weights="DEFAULT")
model.eval()
model.to(device)

# Remove classifier head
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
feature_extractor.to(device)

# --------------------------------------------
# Preprocess
# --------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

image = Image.open("mobile/cat.jpeg").convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

# --------------------------------------------
# Forward
# --------------------------------------------
with torch.no_grad():
    features = feature_extractor(input_tensor)

features = features.view(features.size(0), -1)

print("Feature vector shape:", features.shape)