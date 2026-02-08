"""
Vision-Language assistive inference with live camera feed.

- Shows webcam feed continuously
- User types a question in terminal
- On ENTER:
    - Captures current frame
    - Answers the user's question about the image (English)
    - Translates English → Sinhala
    - Sends Sinhala text to local TTS HTTP server
    - Receives WAV audio and auto-plays it
"""

import os
import cv2
import torch
import tempfile
import subprocess
import json
import urllib.request
import sys
import select
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from qwen_vl_utils import process_vision_info
from translate_en_to_si import translate_text

# -------------------------------------------------------------------
# Environment safety
# -------------------------------------------------------------------
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()
torch.set_grad_enabled(False)

# -------------------------------------------------------------------
# Quantization config
# -------------------------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# -------------------------------------------------------------------
# Load Qwen-VL model
# -------------------------------------------------------------------
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "models/Qwen2.5-VL-3B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.float16,
)
model.eval()

# -------------------------------------------------------------------
# Processor
# -------------------------------------------------------------------
processor = AutoProcessor.from_pretrained(
    "models/Qwen2.5-VL-3B-Instruct",
    min_pixels=256 * 28 * 28,
    max_pixels=1024 * 28 * 28,
    use_fast=True,
)

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
TTS_URL = "http://localhost:8000/tts"
API_KEY = os.getenv("GOOGLE_TRANSLATE_API_KEY")

if not API_KEY:
    raise RuntimeError("GOOGLE_TRANSLATE_API_KEY not found in environment")

os.makedirs("logs", exist_ok=True)

# -------------------------------------------------------------------
# Camera
# -------------------------------------------------------------------
cap = cv2.VideoCapture(0)

print("\nType a question and press ENTER to analyze the image.")
print("Press Q in the camera window to quit.\n")

input_buffer = ""

# -------------------------------------------------------------------
# Camera loop
# -------------------------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Assistive Camera Feed", frame)

    # --------------------------------------------------
    # Handle quit from camera window
    # --------------------------------------------------
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # --------------------------------------------------
    # Non-blocking terminal input
    # --------------------------------------------------
    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        char = sys.stdin.read(1)

        # ENTER → run inference
        if char == "\n":
            question = input_buffer.strip()
            input_buffer = ""

            if not question:
                continue

            print(f"\n[QUESTION] {question}")
            print("[INFO] Capturing frame and running inference...")

            # --------------------------------------------------
            # Capture image
            # --------------------------------------------------
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            fd, temp_image_path = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            pil_image.save(temp_image_path)

            # --------------------------------------------------
            # Build multimodal message
            # --------------------------------------------------
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": temp_image_path},
                        {
                            "type": "text",
                            "text": (
                                "Answer the user's question about the image. "
                                "Be concise, factual, and strictly in English.\n\n"
                                f"Question: {question}"
                            ),
                        },
                    ],
                }
            ]

            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda")

            # --------------------------------------------------
            # Model inference
            # --------------------------------------------------
            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=96,
                    do_sample=False,
                    use_cache=True,
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            english_answer = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            print("English:", english_answer)

            # --------------------------------------------------
            # Translate EN → SI
            # --------------------------------------------------
            sinhala_text = translate_text(english_answer, API_KEY)
            print("Sinhala:", sinhala_text)

            # --------------------------------------------------
            # Send to TTS server
            # --------------------------------------------------
            audio_path = "logs/output.wav"

            payload = json.dumps({"text": sinhala_text}).encode("utf-8")
            req = urllib.request.Request(
                TTS_URL,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(req) as resp:
                audio_data = resp.read()

            with open(audio_path, "wb") as f:
                f.write(audio_data)

            # --------------------------------------------------
            # Auto-play
            # --------------------------------------------------
            if subprocess.call(["which", "aplay"], stdout=subprocess.DEVNULL) == 0:
                subprocess.run(["aplay", audio_path])
            elif subprocess.call(["which", "afplay"], stdout=subprocess.DEVNULL) == 0:
                subprocess.run(["afplay", audio_path])

            os.remove(temp_image_path)
            print("\nType next question:")

        else:
            input_buffer += char

# -------------------------------------------------------------------
# Cleanup
# -------------------------------------------------------------------
cap.release()
cv2.destroyAllWindows()