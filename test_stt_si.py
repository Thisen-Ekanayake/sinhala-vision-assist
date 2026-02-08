import base64
import json
import requests
from dotenv import load_dotenv
import os
import wave
import contextlib
import time
import io
import audioop

load_dotenv()

API_KEY = os.getenv("GOOGLE_STT_API_KEY")
assert API_KEY, "GOOGLE_STT_API_KEY not found"

AUDIO_FILE = "sinhala_stt.wav"

# Inspect WAV to determine sample rate and duration
with contextlib.closing(wave.open(AUDIO_FILE, 'rb')) as wf:
    sample_rate = wf.getframerate()
    channels = wf.getnchannels()
    sampwidth = wf.getsampwidth()
    frames = wf.getnframes()
    duration = frames / float(sample_rate)

# Determine encoding (assume LINEAR16 for common WAV files)
encoding = "LINEAR16" if sampwidth == 2 else "ENCODING_UNSPECIFIED"

# Read audio bytes; convert to mono if necessary
with contextlib.closing(wave.open(AUDIO_FILE, 'rb')) as wf:
    frames = wf.readframes(wf.getnframes())
    if channels > 1:
        # convert stereo -> mono by averaging channels
        mono_frames = audioop.tomono(frames, sampwidth, 0.5, 0.5)
        bio = io.BytesIO()
        with wave.open(bio, 'wb') as outw:
            outw.setnchannels(1)
            outw.setsampwidth(sampwidth)
            outw.setframerate(sample_rate)
            outw.writeframes(mono_frames)
        audio_bytes = bio.getvalue()
    else:
        # read raw file bytes
        with open(AUDIO_FILE, 'rb') as f:
            audio_bytes = f.read()

audio_content = base64.b64encode(audio_bytes).decode('utf-8')

config = {
    "encoding": encoding,
    "sampleRateHertz": sample_rate,
    "languageCode": "si-LK",
    "model": "default",
}

payload = {
    "config": config,
    "audio": {"content": audio_content},
}

def post_sync(payload):
    url = f"https://speech.googleapis.com/v1/speech:recognize?key={API_KEY}"
    resp = requests.post(url, json=payload)
    if not resp.ok:
        print("Request failed:", resp.status_code, resp.text)
        resp.raise_for_status()
    return resp.json()

def post_longrunning(payload, timeout=300, poll_interval=2):
    url = f"https://speech.googleapis.com/v1/speech:longrunningrecognize?key={API_KEY}"
    resp = requests.post(url, json=payload)
    if not resp.ok:
        print("Longrunning request failed:", resp.status_code, resp.text)
        resp.raise_for_status()
    op = resp.json()
    name = op.get("name") or op.get("operation")
    if not name:
        # older responses return the full operation object
        return op
    # poll operation
    op_url = f"https://speech.googleapis.com/v1/operations/{name}?key={API_KEY}"
    start = time.time()
    while True:
        r = requests.get(op_url)
        if not r.ok:
            print("Operation poll failed:", r.status_code, r.text)
            r.raise_for_status()
        j = r.json()
        if j.get("done"):
            return j
        if time.time() - start > timeout:
            raise TimeoutError("Longrunning recognition timed out")
        time.sleep(poll_interval)


if duration > 60:
    print(f"Audio duration {duration:.1f}s > 60s, using longrunningrecognize")
    result = post_longrunning(payload)
else:
    result = post_sync(payload)

print("\n--- Transcription ---")
for r in result.get("results", []):
    print(r["alternatives"][0]["transcript"])