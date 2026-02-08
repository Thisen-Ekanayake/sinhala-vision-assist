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
import sys
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
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

from qwen_vl_utils import process_vision_info
from translate_en_to_si import translate_text  # EN -> SI helper


# -------------------------------------------------------------------
# Env / keys
# -------------------------------------------------------------------
load_dotenv()

GOOGLE_STT_API_KEY = os.getenv("GOOGLE_STT_API_KEY")
GOOGLE_TRANSLATE_API_KEY = os.getenv("GOOGLE_TRANSLATE_API_KEY")
GOOGLE_TTS_API_KEY = os.getenv("GOOGLE_TTS_API_KEY")

if not GOOGLE_STT_API_KEY:
    raise RuntimeError("GOOGLE_STT_API_KEY not found (.env)")
if not GOOGLE_TRANSLATE_API_KEY:
    raise RuntimeError("GOOGLE_TRANSLATE_API_KEY not found (.env)")
if not GOOGLE_TTS_API_KEY:
    raise RuntimeError("GOOGLE_TTS_API_KEY not found (.env)")

os.makedirs("logs", exist_ok=True)

# -------------------------------------------------------------------
# Torch safety
# -------------------------------------------------------------------
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.cuda.empty_cache()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------------------
# Load Qwen-VL (same idea as your current cam_vl.py)
# -------------------------------------------------------------------
# NOTE: bitsandbytes 4-bit quantization generally expects CUDA.
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

# -------------------------------------------------------------------
# Google Speech (STT) – based on stt_si.py :contentReference[oaicite:2]{index=2}
# -------------------------------------------------------------------
STT_SAMPLE_RATE = 16000
STT_CHANNELS = 1
STT_DTYPE = "int16"
STT_LANGUAGE_CODE = "si-LK"


def google_stt_recognize(audio_np_int16: np.ndarray) -> str:
    """Return best transcript (Sinhala) or empty string."""
    audio_bytes = audio_np_int16.tobytes()
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    payload = {
        "config": {
            "encoding": "LINEAR16",
            "sampleRateHertz": STT_SAMPLE_RATE,
            "languageCode": STT_LANGUAGE_CODE,
            "model": "default",
        },
        "audio": {"content": audio_b64},
    }

    url = f"https://speech.googleapis.com/v1/speech:recognize?key={GOOGLE_STT_API_KEY}"
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    # pick the first alternative transcript if present
    results = data.get("results", [])
    if not results:
        return ""
    alts = results[0].get("alternatives", [])
    if not alts:
        return ""
    return (alts[0].get("transcript") or "").strip()


# -------------------------------------------------------------------
# Google Translate v2 REST
# - translate_en_to_si.py already does EN->SI :contentReference[oaicite:3]{index=3}
# - we add SI->EN here (same API/key)
# -------------------------------------------------------------------
def translate_rest(text: str, source: str, target: str, api_key: str) -> str:
    url = "https://translation.googleapis.com/language/translate/v2"
    data = {
        "q": text,
        "source": source,
        "target": target,
        "format": "text",
        "key": api_key,
    }
    encoded = urllib.parse.urlencode(data).encode("utf-8")
    req = urllib.request.Request(url, data=encoded, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")

    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read().decode("utf-8")
        payload = json.loads(body)

    try:
        return payload["data"]["translations"][0]["translatedText"].strip()
    except Exception as e:
        raise RuntimeError(f"Unexpected response from translate API: {payload}") from e


# -------------------------------------------------------------------
# Google TTS – based on tts_si.py :contentReference[oaicite:4]{index=4}
# -------------------------------------------------------------------
TTS_LANGUAGE_CODE = "si-LK"
TTS_AUDIO_ENCODING = "LINEAR16"


def google_tts_synthesize_si(text_si: str) -> bytes:
    url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={GOOGLE_TTS_API_KEY}"
    payload = {
        "input": {"text": text_si},
        "voice": {"languageCode": TTS_LANGUAGE_CODE, "ssmlGender": "NEUTRAL"},
        "audioConfig": {"audioEncoding": TTS_AUDIO_ENCODING},
    }
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    audio_b64 = resp.json()["audioContent"]
    return base64.b64decode(audio_b64)


def autoplay_wav(path: str) -> None:
    # Linux: aplay, macOS: afplay
    if subprocess.call(["which", "aplay"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
        subprocess.run(["aplay", path], check=False)
    elif subprocess.call(["which", "afplay"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
        subprocess.run(["afplay", path], check=False)


# -------------------------------------------------------------------
# Audio recorder (toggle with 'r')
# -------------------------------------------------------------------
class ToggleRecorder:
    def __init__(self, samplerate=STT_SAMPLE_RATE, channels=STT_CHANNELS, dtype=STT_DTYPE):
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self.recording = False
        self._q: "queue.Queue[np.ndarray]" = queue.Queue()
        self._frames: list[np.ndarray] = []

        def _cb(indata, frames, time_info, status):
            if self.recording:
                self._q.put(indata.copy())

        self._stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype=self.dtype,
            callback=_cb,
        )

    def __enter__(self):
        self._stream.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self._stream.__exit__(exc_type, exc, tb)

    def start(self):
        self._frames = []
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self.recording = True

    def stop_and_collect(self) -> np.ndarray | None:
        self.recording = False
        # drain queue
        while True:
            try:
                chunk = self._q.get_nowait()
                self._frames.append(chunk)
            except queue.Empty:
                break

        if not self._frames:
            return None
        audio = np.concatenate(self._frames, axis=0)
        # ensure int16 contiguous
        if audio.dtype != np.int16:
            audio = audio.astype(np.int16, copy=False)
        return np.ascontiguousarray(audio)


# -------------------------------------------------------------------
# Camera loop
# -------------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    print("\n[INFO] Camera running.")
    print("[INFO] Press 'r' in the camera window to START recording Sinhala.")
    print("[INFO] Press 'r' again to STOP, then it will run STT->Translate->VL->Translate->TTS.")
    print("[INFO] Press 'q' to quit.\n")

    last_frame = None
    recording_started_at = None

    with ToggleRecorder() as rec:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            last_frame = frame
            cv2.imshow("Assistive Camera Feed", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r"):
                # toggle record
                if not rec.recording:
                    rec.start()
                    recording_started_at = time.time()
                    print("[REC] Recording started... (press 'r' again to stop)")
                else:
                    audio_np = rec.stop_and_collect()
                    dur = (time.time() - (recording_started_at or time.time()))
                    print(f"[REC] Recording stopped ({dur:.2f}s)")

                    if audio_np is None or audio_np.size == 0:
                        print("[WARN] No audio captured.")
                        continue

                    # -------------------------
                    # STT: Sinhala
                    # -------------------------
                    print("[STT] Transcribing Sinhala...")
                    si_text = google_stt_recognize(audio_np)
                    if not si_text:
                        print("[WARN] No transcription returned.")
                        continue
                    print(f"[SI] {si_text}")

                    # -------------------------
                    # Translate SI -> EN
                    # -------------------------
                    print("[MT] Translating Sinhala -> English...")
                    en_question = translate_rest(si_text, source="si", target="en", api_key=GOOGLE_TRANSLATE_API_KEY)
                    print(f"[EN question] {en_question}")

                    # -------------------------
                    # Capture current frame for VL
                    # -------------------------
                    if last_frame is None:
                        print("[WARN] No frame available.")
                        continue

                    rgb = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb)
                    fd, temp_image_path = tempfile.mkstemp(suffix=".jpg")
                    os.close(fd)
                    pil_image.save(temp_image_path)

                    try:
                        # -------------------------
                        # VL prompt (English)
                        # -------------------------
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
                                            f"Question: {en_question}"
                                        ),
                                    },
                                ],
                            }
                        ]

                        chat_text = processor.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        image_inputs, video_inputs = process_vision_info(messages)

                        inputs = processor(
                            text=[chat_text],
                            images=image_inputs,
                            videos=video_inputs,
                            padding=True,
                            return_tensors="pt",
                        )

                        if DEVICE == "cuda":
                            inputs = inputs.to("cuda")

                        # -------------------------
                        # Inference
                        # -------------------------
                        print("[VL] Running model...")
                        with torch.inference_mode():
                            generated_ids = model.generate(
                                **inputs,
                                max_new_tokens=128,
                                do_sample=False,
                                use_cache=True,
                            )

                        trimmed = [
                            out_ids[len(in_ids):]
                            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                        ]
                        en_answer = processor.batch_decode(
                            trimmed,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )[0].strip()

                        print(f"[EN answer] {en_answer}")

                        # -------------------------
                        # Translate EN -> SI (reuse translate_en_to_si.py)
                        # -------------------------
                        print("[MT] Translating English -> Sinhala...")
                        si_answer = translate_text(en_answer, GOOGLE_TRANSLATE_API_KEY)
                        print(f"[SI answer] {si_answer}")

                        # -------------------------
                        # TTS (Sinhala) -> wav bytes
                        # -------------------------
                        print("[TTS] Synthesizing Sinhala audio...")
                        audio_bytes = google_tts_synthesize_si(si_answer)

                        out_path = os.path.join("logs", "output.wav")
                        with open(out_path, "wb") as f:
                            f.write(audio_bytes)

                        print(f"[OK] Saved: {out_path}")
                        autoplay_wav(out_path)

                    finally:
                        try:
                            os.remove(temp_image_path)
                        except Exception:
                            pass

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()