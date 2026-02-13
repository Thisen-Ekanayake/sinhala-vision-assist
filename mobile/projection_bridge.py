import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_large
from PIL import Image

# --------------------------------------------
# Setup
# --------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

mobilenet = mobilenet_v3_large(weights="DEFAULT")
mobilenet.eval()
mobilenet.to(device)

feature_extractor = torch.nn.Sequential(*list(mobilenet.children())[:-1])

# Assume LLM hidden size = 768 (example)
LLM_HIDDEN_SIZE = 768

projection = torch.nn.Linear(960, LLM_HIDDEN_SIZE).to(device)

# --------------------------------------------
# Preprocess
# --------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

image = Image.open("mobile/cat.jpeg").convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

# --------------------------------------------
# Forward
# --------------------------------------------
with torch.no_grad():
    features = feature_extractor(input_tensor)
    features = features.view(features.size(0), -1)

projected = projection(features)

print("Original feature shape:", features.shape)
print("Projected feature shape:", projected.shape)