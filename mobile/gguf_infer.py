import time
from llama_cpp import Llama

# path to your GGUF model
model_path = "models/qwen2.5-0.5b-f16.gguf"

# initialize the model
llm = Llama(
    model_path=model_path,
    n_ctx=2048,        # context size
    n_threads=8        # CPU threads
)

# prompt to run
prompt = "Explain quantum physics in simple terms"

# start timing
start = time.perf_counter()

# model call
output = llm(
    prompt,
    max_tokens=100,
    temperature=0.7,
    top_p=0.9
)

# end timing
end = time.perf_counter()

# extract generated text
result_text = output["choices"][0]["text"].strip()

# compute duration
duration_secs = end - start

# print results
print(f"Result:\n{result_text}\n")
print(f"Inference time (seconds): {duration_secs:.4f}")