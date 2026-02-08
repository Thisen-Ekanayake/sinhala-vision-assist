"""
Vision-Language assistive inference with live camera feed + Sinhala voice input.

Flow (single key: r):
- Shows webcam feed continuously
- Press 'r' in the camera window to START recording Sinhala
- Press 'r' again to STOP recording
- STT (si-LK) -> Sinhala transcript (Google Speech REST, same as stt_si.py)
- Translate Sinhala -> English (Google Translate v2 REST)
- Capture current frame
- Ask the VL model in English using (translated) question + image
- Model outputs English answer
- Translate English -> Sinhala (translate_en_to_si.py)
- TTS Sinhala -> WAV bytes (tts_si.py logic)
- Save + auto-play

Keys:
- r : toggle record start/stop (and run the full pipeline after stop)
- q : quit
"""

import os
import cv2
import time
import json
import base64
import queue
import tempfile
import subprocess
import urllib.parse
import urllib.request

import numpy as np
import sounddevice as sd
import requests
import torch
from dotenv import load_dotenv
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)

from qwen_vl_utils import process_vision_info
from translate_en_to_si import translate_text  # EN -> SI


# ---------------------------------------------------------------------
# Environment / API keys
# ---------------------------------------------------------------------
load_dotenv()

GOOGLE_STT_API_KEY = os.getenv("GOOGLE_STT_API_KEY")
GOOGLE_TRANSLATE_API_KEY = os.getenv("GOOGLE_TRANSLATE_API_KEY")
GOOGLE_TTS_API_KEY = os.getenv("GOOGLE_TTS_API_KEY")

if not all([GOOGLE_STT_API_KEY, GOOGLE_TRANSLATE_API_KEY, GOOGLE_TTS_API_KEY]):
    raise RuntimeError("Missing Google API keys in .env")

os.makedirs("logs", exist_ok=True)

# ---------------------------------------------------------------------
# Torch safety
# ---------------------------------------------------------------------
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.cuda.empty_cache()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------
# Load Qwen2.5-VL
# ---------------------------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

MODEL_PATH = "models/Qwen2.5-VL-3B-Instruct"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config if DEVICE == "cuda" else None,
    device_map="auto" if DEVICE == "cuda" else None,
    dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
)
model.eval()

processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    min_pixels=256 * 28 * 28,
    max_pixels=1024 * 28 * 28,
    use_fast=True,
)

# ---------------------------------------------------------------------
# STT config
# ---------------------------------------------------------------------
STT_SAMPLE_RATE = 16000
STT_CHANNELS = 1
STT_DTYPE = "int16"
STT_LANGUAGE = "si-LK"


def google_stt(audio_np: np.ndarray) -> str:
    audio_b64 = base64.b64encode(audio_np.tobytes()).decode()
    payload = {
        "config": {
            "encoding": "LINEAR16",
            "sampleRateHertz": STT_SAMPLE_RATE,
            "languageCode": STT_LANGUAGE,
        },
        "audio": {"content": audio_b64},
    }
    url = f"https://speech.googleapis.com/v1/speech:recognize?key={GOOGLE_STT_API_KEY}"
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    res = r.json().get("results", [])
    if not res:
        return ""
    return res[0]["alternatives"][0]["transcript"].strip()


# ---------------------------------------------------------------------
# Translate (SI -> EN)
# ---------------------------------------------------------------------
def translate_si_to_en(text: str) -> str:
    url = "https://translation.googleapis.com/language/translate/v2"
    data = {
        "q": text,
        "source": "si",
        "target": "en",
        "format": "text",
        "key": GOOGLE_TRANSLATE_API_KEY,
    }
    encoded = urllib.parse.urlencode(data).encode()
    req = urllib.request.Request(url, data=encoded, method="POST")
    with urllib.request.urlopen(req) as resp:
        payload = json.loads(resp.read())
    return payload["data"]["translations"][0]["translatedText"].strip()


# ---------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------
def google_tts_si(text_si: str) -> bytes:
    url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={GOOGLE_TTS_API_KEY}"
    payload = {
        "input": {"text": text_si},
        "voice": {"languageCode": "si-LK", "ssmlGender": "NEUTRAL"},
        "audioConfig": {"audioEncoding": "LINEAR16"},
    }
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    return base64.b64decode(r.json()["audioContent"])


def play_wav(path: str):
    if subprocess.call(["which", "aplay"], stdout=subprocess.DEVNULL) == 0:
        subprocess.run(["aplay", path], check=False)


# ---------------------------------------------------------------------
# Audio recorder (toggle with r)
# ---------------------------------------------------------------------
class Recorder:
    def __init__(self):
        self.recording = False
        self.q = queue.Queue()
        self.frames = []

        def callback(indata, frames, time_info, status):
            if self.recording:
                self.q.put(indata.copy())

        self.stream = sd.InputStream(
            samplerate=STT_SAMPLE_RATE,
            channels=STT_CHANNELS,
            dtype=STT_DTYPE,
            callback=callback,
        )

    def __enter__(self):
        self.stream.__enter__()
        return self

    def __exit__(self, *args):
        self.stream.__exit__(*args)

    def start(self):
        self.frames.clear()
        while not self.q.empty():
            self.q.get_nowait()
        self.recording = True

    def stop(self):
        self.recording = False
        while not self.q.empty():
            self.frames.append(self.q.get_nowait())
        if not self.frames:
            return None
        return np.concatenate(self.frames, axis=0)


# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam not available")

    print("\n[r] start/stop recording | [q] quit\n")

    pipeline_start_time = None
    last_frame = None

    with Recorder() as rec:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            last_frame = frame
            cv2.imshow("Assistive Vision", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r"):
                if not rec.recording:
                    pipeline_start_time = time.time()
                    rec.start()
                    print("[REC] started")
                else:
                    audio = rec.stop()
                    print("[REC] stopped")

                    if audio is None:
                        print("[WARN] no audio")
                        continue

                    # ---- STT
                    print("[STT]")
                    si_text = google_stt(audio)
                    print("SI:", si_text)

                    # ---- Translate SI -> EN
                    print("[MT] si -> en")
                    en_q = translate_si_to_en(si_text)
                    print("EN:", en_q)

                    # ---- VL
                    rgb = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(rgb)
                    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                    img.save(tmp.name)

                    try:
                        messages = [{
                            "role": "user",
                            "content": [
                                {"type": "image", "image": tmp.name},
                                {"type": "text", "text": f"Answer the question:\n{en_q}"}
                            ],
                        }]

                        text = processor.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        imgs, vids = process_vision_info(messages)
                        inputs = processor(
                            text=[text],
                            images=imgs,
                            videos=vids,
                            return_tensors="pt",
                        )
                        if DEVICE == "cuda":
                            inputs = inputs.to("cuda")

                        print("[VLM]")
                        with torch.inference_mode():
                            out = model.generate(**inputs, max_new_tokens=128)

                        gen = out[0][inputs.input_ids.shape[1]:]
                        en_ans = processor.decode(gen, skip_special_tokens=True)
                        print("EN answer:", en_ans)

                        # ---- Translate EN -> SI
                        print("[MT] en -> si")
                        si_ans = translate_text(en_ans, GOOGLE_TRANSLATE_API_KEY)
                        print("SI answer:", si_ans)

                        # ---- TTS
                        print("[TTS]")
                        audio_bytes = google_tts_si(si_ans)
                        out_path = "logs/output.wav"
                        with open(out_path, "wb") as f:
                            f.write(audio_bytes)

                        # ---- Latency
                        total_time = time.time() - pipeline_start_time
                        print(f"[LATENCY] end-to-end: {total_time:.2f}s")

                        play_wav(out_path)

                    finally:
                        os.unlink(tmp.name)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()