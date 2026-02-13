import time
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_large
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

# --------------------------------------------------
# Config
# --------------------------------------------------
IMAGE_PATH = "mobile/cat.jpeg"
QWEN_PATH = "models/Qwen2.5-0.5B-Instruct"  # HF version

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# --------------------------------------------------
# Load MobileNet
# --------------------------------------------------
print("\nLoading MobileNet...")
mobilenet = mobilenet_v3_large(weights="DEFAULT")
mobilenet.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --------------------------------------------------
# Load Qwen
# --------------------------------------------------
print("Loading Qwen...")
tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH)
qwen = AutoModelForCausalLM.from_pretrained(
    QWEN_PATH,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).to(device)
qwen.eval()

# --------------------------------------------------
# Load labels locally
# --------------------------------------------------
with open("mobile/imagenet_classes.txt", "r") as f:
    labels = [line.strip() for line in f]

# --------------------------------------------------
# Begin Total Timer
# --------------------------------------------------
total_start = time.time()

# --------------------------------------------------
# STEP 1 — Vision
# --------------------------------------------------
vision_start = time.time()

image = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = mobilenet(input_tensor)

probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
top5_prob, top5_catid = torch.topk(probabilities, 5)

# Create list of (label, confidence) tuples
top_labels = [(labels[i.item()], prob.item()) for i, prob in zip(top5_catid, top5_prob)]

vision_end = time.time()
print("\nVision output with confidence scores:")
for label, conf in top_labels:
    print(f"  {label}: {conf:.1%}")
print("Vision time:", vision_end - vision_start, "seconds")

# --------------------------------------------------
# STEP 2 — Prompt Construction
# --------------------------------------------------
prompt_start = time.time()

# Get only high-confidence predictions
high_conf_labels = [(label, score) for label, score in top_labels if score > 0.5]

if not high_conf_labels:
    # If nothing above 50%, take top 2
    high_conf_labels = top_labels[:2]

vision_text = ", ".join([f"{label} ({score:.0%})" for label, score in high_conf_labels])

print(f"\nHigh confidence predictions: {vision_text}")

# Use chat template
messages = [
    {"role": "system", "content": "You describe images accurately based on vision model predictions."},
    {"role": "user", "content": f"Computer vision detected: {vision_text}. Briefly describe what's in the image."}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

prompt_end = time.time()
print("Prompt construction time:", prompt_end - prompt_start, "seconds")
print("\nGenerated prompt:")
print(prompt)
print("\n" + "="*60)

# --------------------------------------------------
# STEP 3 — Qwen Generation
# --------------------------------------------------
llm_start = time.time()

inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = qwen.generate(
        **inputs,
        max_new_tokens=60,
        do_sample=True,
        temperature=0.3,
        top_p=0.9,
        repetition_penalty=1.1
    )

generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

llm_end = time.time()
print("\nQwen output:\n")
print(generated)
print("\n" + "="*60)

print("\nLLM generation time:", llm_end - llm_start, "seconds")

# --------------------------------------------------
# TOTAL TIME
# --------------------------------------------------
total_end = time.time()
print("\nTotal pipeline time:", total_end - total_start, "seconds")