"""
Vision-Language assistive inference with live camera feed.

- Shows webcam feed
- Captures a frame on SPACE
- Generates English narration using Qwen2.5-VL
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
# Assistive prompt (FORCE ENGLISH OUTPUT)
# -------------------------------------------------------------------
assistive_prompt = (
    "You are an assistive vision system for a blind user. "
    "Describe the most important objects or actions in front of the user. "
    "Mention obstacles, people, or movement if present. "
    "Keep it brief, practical, and strictly in English."
)

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
TTS_URL = "http://localhost:8000/tts"
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not found in environment")

os.makedirs("logs", exist_ok=True)

# -------------------------------------------------------------------
# Camera loop
# -------------------------------------------------------------------
cap = cv2.VideoCapture(0)

print("Press SPACE to narrate | Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Assistive Camera Feed", frame)
    key = cv2.waitKey(1) & 0xFF

    # Quit
    if key == ord("q"):
        break

    # Capture + narrate
    if key == ord(" "):
        print("\n[INFO] Capturing frame...")

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        fd, temp_image_path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        pil_image.save(temp_image_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": temp_image_path},
                    {"type": "text", "text": assistive_prompt},
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

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False,
                use_cache=True,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        english_narration = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        print("English:", english_narration)

        # --------------------------------------------------
        # Translate EN → SI
        # --------------------------------------------------
        sinhala_text = translate_text(english_narration, API_KEY)
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

# -------------------------------------------------------------------
# Cleanup
# -------------------------------------------------------------------
cap.release()
cv2.destroyAllWindows()