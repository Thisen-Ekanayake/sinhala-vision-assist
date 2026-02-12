import time
from llama_cpp import Llama

# path to your quantized GGUF model
model_path = "models/qwen2.5-0.5b-q4_k_m.gguf"

# initialize the quantized model
llm = Llama(
    model_path=model_path,
    n_ctx=2048,        # context/window size
    n_threads=8        # threads used for CPU inference
)

# the prompt you want to run
prompt = "Explain quantum physics in simple terms"

# start timing just before inference
start = time.perf_counter()

# model call (text generation)
output = llm(
    prompt,
    max_tokens=50,
    temperature=0.7,
    top_p=0.9
)

# end timing after inference
end = time.perf_counter()

# extract the generated text
result_text = output["choices"][0]["text"].strip()

# compute the inference duration
duration_secs = end - start

# print results
print(f"Result:\n{result_text}\n")
print(f"Inference time (seconds): {duration_secs:.4f}")