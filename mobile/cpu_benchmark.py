import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_large
from PIL import Image
import time

device = "cpu"
print("Running on CPU")

model = mobilenet_v3_large(weights="DEFAULT")
model.eval()
model.to(device)

feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

image = Image.open("mobile/cat.jpeg").convert("RGB")
input_tensor = transform(image).unsqueeze(0)

# Warmup
for _ in range(3):
    with torch.no_grad():
        feature_extractor(input_tensor)

start = time.time()
with torch.no_grad():
    feature_extractor(input_tensor)
end = time.time()

print("CPU inference time:", end - start, "seconds")