import os
import base64
import requests
from dotenv import load_dotenv

class SinhalaTTS:
    """
    Sinhala Text-to-Speech using Google Text-to-Speech REST API
    """

    LANGUAGE_CODE = "si-LK"
    AUDIO_ENCODING = "LINEAR16"

    def __init__(self, output_path: str = "sinhala_tts.wav"):
        # -------------------------
        # Load environment
        # -------------------------
        load_dotenv()
        self.api_key = os.getenv("GOOGLE_TTS_API_KEY")
        assert self.api_key, "GOOGLE_TTS_API_KEY not found"

        self.output_path = output_path
        self.url = (
            "https://texttospeech.googleapis.com/v1/text:synthesize"
            f"?key={self.api_key}"
        )

    # -------------------------
    # Core TTS
    # -------------------------
    def synthesize(self, text: str) -> bytes:
        payload = {
            "input": {
                "text": text
            },
            "voice": {
                "languageCode": self.LANGUAGE_CODE,
                "ssmlGender": "NEUTRAL"
            },
            "audioConfig": {
                "audioEncoding": self.AUDIO_ENCODING
            }
        }

        resp = requests.post(self.url, json=payload)
        resp.raise_for_status()

        audio_b64 = resp.json()["audioContent"]
        return base64.b64decode(audio_b64)

    # -------------------------
    # Callable execution
    # -------------------------
    def __call__(self):
        print("\nEnter Sinhala text to convert to speech:")
        text = input(">> ").strip()

        if not text:
            print("No text provided. Exiting.")
            return

        audio_bytes = self.synthesize(text)

        with open(self.output_path, "wb") as f:
            f.write(audio_bytes)

        print(f"\n[OK] Audio saved to: {self.output_path}")


# --------------------------------------------------
# Executable entry point
# --------------------------------------------------
if __name__ == "__main__":
    print("Sinhala Text-to-Speech (Google REST)\n")

    tts = SinhalaTTS()
    tts()