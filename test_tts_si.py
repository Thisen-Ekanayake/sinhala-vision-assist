import requests
import base64
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("GOOGLE_TTS_API_KEY")
assert API_KEY, "GOOGLE_TTS_API_KEY not found"

url = (
    "https://texttospeech.googleapis.com/v1/text:synthesize"
    f"?key={API_KEY}"
)

payload = {
    "input": {
        "text": "ආයුබෝවන්, ඔබට කොහොමද?"
    },
    "voice": {
        "languageCode": "si-LK",
        "ssmlGender": "NEUTRAL"
    },
    "audioConfig": {
        "audioEncoding": "LINEAR16"
    }
}

resp = requests.post(url, json=payload)
resp.raise_for_status()

audio_b64 = resp.json()["audioContent"]
audio_bytes = base64.b64decode(audio_b64)

with open("sinhala_tts.wav", "wb") as f:
    f.write(audio_bytes)

print("Saved sinhala_tts.wav")