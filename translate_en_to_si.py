"""
English to Sinhala Translation Script
Uses Google Cloud Translation API
"""

import os
import sys
from dotenv import load_dotenv
import json
import urllib.parse
import urllib.request

# try importing google-cloud client for alternate credentialed usage
try:
    from google.cloud import translate_v2 as translate  # type: ignore
except Exception:
    translate = None

# Load variables from .env into environment
load_dotenv()


def translate_text(text, api_key):
    """
    Translate English text to Sinhala using Google Cloud Translation API
    """
    # If we have an API key string, call the REST v2 endpoint (works with API key)
    if api_key:
        url = "https://translation.googleapis.com/language/translate/v2"
        data = {
            "q": text,
            "source": "en",
            "target": "si",
            "format": "text",
            "key": api_key,
        }

        encoded = urllib.parse.urlencode(data).encode("utf-8")
        req = urllib.request.Request(url, data=encoded, method="POST")
        req.add_header("Content-Type", "application/x-www-form-urlencoded")

        with urllib.request.urlopen(req) as resp:
            body = resp.read().decode("utf-8")
            payload = json.loads(body)

        # expected structure: {"data": {"translations": [{"translatedText": "..."}]}}
        try:
            return payload["data"]["translations"][0]["translatedText"]
        except Exception as e:
            raise RuntimeError(f"Unexpected response from translate API: {payload}") from e

    # Fallback: use google.cloud client library if available and credentials are configured
    if translate is None:
        raise RuntimeError("No translation client available and no API key provided")

    client = translate.Client()
    result = client.translate(text, source_language="en", target_language="si")
    return result["translatedText"]


def main():
    api_key = os.getenv("GOOGLE_TRANSLATE_API_KEY")

    if not api_key:
        print("❌ ERROR: GOOGLE_TRANSLATE_API_KEY not found.")
        print("➡️  Make sure you have a .env file with GOOGLE_TRANSLATE_API_KEY set.")
        sys.exit(1)

    print("English → Sinhala Translator")
    print("=" * 40)
    print("Type 'quit' or 'exit' to stop\n")

    while True:
        english_text = input("Enter English text: ").strip()

        if english_text.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        if not english_text:
            print("Please enter some text.\n")
            continue

        try:
            sinhala_text = translate_text(english_text, api_key)
            print(f"Sinhala: {sinhala_text}\n")

        except Exception as e:
            print(f"Error: {e}")
            print("Check API key, quota, or network.\n")


if __name__ == "__main__":
    main()