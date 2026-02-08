import os
import sys
import time
import json
import base64
import queue
import threading
import requests
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv

# --------------------------------------------------
# ENV
# --------------------------------------------------
load_dotenv()
API_KEY = os.getenv("GOOGLE_STT_API_KEY")
assert API_KEY, "GOOGLE_STT_API_KEY not found"

# --------------------------------------------------
# Audio config
# --------------------------------------------------
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"

audio_queue = queue.Queue()
recording = False
frames = []

# --------------------------------------------------
# Audio callback
# --------------------------------------------------
def audio_callback(indata, frames_count, time_info, status):
    if recording:
        audio_queue.put(indata.copy())

# --------------------------------------------------
# Keyboard listener
# --------------------------------------------------
def keyboard_listener():
    global recording, frames
    print("\nPress 'r' to record | 's' to stop | Ctrl+C to quit\n")

    while True:
        key = sys.stdin.read(1)

        if key == "r" and not recording:
            frames = []
            recording = True
            print("[REC] Recording started...")

        elif key == "s" and recording:
            recording = False
            print("[REC] Recording stopped")
            break

# --------------------------------------------------
# STT request
# --------------------------------------------------
def recognize_audio(audio_np):
    audio_bytes = audio_np.tobytes()
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    payload = {
        "config": {
            "encoding": "LINEAR16",
            "sampleRateHertz": SAMPLE_RATE,
            "languageCode": "si-LK",
            "model": "default",
        },
        "audio": {
            "content": audio_b64
        }
    }

    url = f"https://speech.googleapis.com/v1/speech:recognize?key={API_KEY}"
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    return resp.json()

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    global frames

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        callback=audio_callback,
    )

    with stream:
        kb_thread = threading.Thread(target=keyboard_listener, daemon=True)
        kb_thread.start()

        while kb_thread.is_alive():
            try:
                chunk = audio_queue.get(timeout=0.1)
                frames.append(chunk)
            except queue.Empty:
                pass

    if not frames:
        print("No audio recorded.")
        return

    audio_np = np.concatenate(frames, axis=0)
    print(f"[INFO] Captured {audio_np.shape[0] / SAMPLE_RATE:.2f}s of audio")

    result = recognize_audio(audio_np)

    print("\n--- Transcription ---")
    for r in result.get("results", []):
        print(r["alternatives"][0]["transcript"])


if __name__ == "__main__":
    print("Live Sinhala STT (Google REST)")
    print("Make sure your mic is working.\n")
    main()