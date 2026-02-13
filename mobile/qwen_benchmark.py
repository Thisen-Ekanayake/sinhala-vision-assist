import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "models/Qwen2.5-0.5B-Instruct"

prompt = "Explain what a neural network is in simple terms."

def run_benchmark(device):
    print("\n==============================")
    print(f"Running on {device.upper()}")
    print("==============================")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Warmup
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=100)

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.time()

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)

    if device == "cuda":
        torch.cuda.synchronize()

    end = time.time()

    # Extract generated tokens only
    generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
    num_generated_tokens = generated_tokens.shape[0]

    decoded_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    total_time = end - start
    tokens_per_sec = num_generated_tokens / total_time

    print(f"Generated tokens: {num_generated_tokens}")
    print(f"Total inference time: {total_time:.4f} seconds")
    print(f"Tokens per second: {tokens_per_sec:.2f}")
    print("\nGenerated text:\n")
    print(decoded_text)


# -------------------------
# Run CPU
# -------------------------
run_benchmark("cpu")

# -------------------------
# Run GPU (if available)
# -------------------------
if torch.cuda.is_available():
    run_benchmark("cuda")
else:
    print("\nCUDA not available. GPU test skipped.")