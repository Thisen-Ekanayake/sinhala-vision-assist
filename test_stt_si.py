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

class LiveSinhalaSTT:
    """
    Live Sinhala Speech-to-Text using Google Speech REST API
    Press:
      r -> start recording
      s -> stop recording
    """

    # -------------------------
    # Audio config (class-level)
    # -------------------------
    SAMPLE_RATE = 16000
    CHANNELS = 1
    DTYPE = "int16"
    LANGUAGE_CODE = "si-LK"

    def __init__(self):
        # -------------------------
        # Load environment
        # -------------------------
        load_dotenv()
        self.api_key = os.getenv("GOOGLE_STT_API_KEY")
        assert self.api_key, "GOOGLE_STT_API_KEY not found"

        # -------------------------
        # Runtime state
        # -------------------------
        self.audio_queue = queue.Queue()
        self.recording = False
        self.frames = []

    # -------------------------
    # Audio callback
    # -------------------------
    def audio_callback(self, indata, frames_count, time_info, status):
        if self.recording:
            self.audio_queue.put(indata.copy())

    # -------------------------
    # Keyboard listener
    # -------------------------
    def keyboard_listener(self):
        print("\nPress 'r' to record | 's' to stop | Ctrl+C to quit\n")

        while True:
            key = sys.stdin.read(1)

            if key == "r" and not self.recording:
                self.frames = []
                self.recording = True
                print("[REC] Recording started...")

            elif key == "s" and self.recording:
                self.recording = False
                print("[REC] Recording stopped")
                break

    # -------------------------
    # STT request
    # -------------------------
    def recognize_audio(self, audio_np: np.ndarray) -> dict:
        audio_bytes = audio_np.tobytes()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        payload = {
            "config": {
                "encoding": "LINEAR16",
                "sampleRateHertz": self.SAMPLE_RATE,
                "languageCode": self.LANGUAGE_CODE,
                "model": "default",
            },
            "audio": {
                "content": audio_b64
            }
        }

        url = f"https://speech.googleapis.com/v1/speech:recognize?key={self.api_key}"
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()

    # -------------------------
    # Main execution (callable)
    # -------------------------
    def __call__(self):
        stream = sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            channels=self.CHANNELS,
            dtype=self.DTYPE,
            callback=self.audio_callback,
        )

        with stream:
            kb_thread = threading.Thread(
                target=self.keyboard_listener,
                daemon=True
            )
            kb_thread.start()

            while kb_thread.is_alive():
                try:
                    chunk = self.audio_queue.get(timeout=0.1)
                    self.frames.append(chunk)
                except queue.Empty:
                    pass

        if not self.frames:
            print("No audio recorded.")
            return

        audio_np = np.concatenate(self.frames, axis=0)
        duration = audio_np.shape[0] / self.SAMPLE_RATE
        print(f"[INFO] Captured {duration:.2f}s of audio")

        result = self.recognize_audio(audio_np)

        print("\n--- Transcription ---")
        for r in result.get("results", []):
            print(r["alternatives"][0]["transcript"])


# --------------------------------------------------
# Executable entry point
# --------------------------------------------------
if __name__ == "__main__":
    print("Live Sinhala STT (Google REST)")
    print("Make sure your mic is working.\n")

    stt = LiveSinhalaSTT()
    stt()
