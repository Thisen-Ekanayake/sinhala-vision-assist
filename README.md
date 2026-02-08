# Object Detection & Sinhala STT — Vision–Language Assistive Pipeline

A Python project that combines **live webcam vision**, **Sinhala speech input**, and a **vision–language model** to answer questions in Sinhala. You speak a question in Sinhala, the system captures the current camera frame, runs a VLM (Qwen2.5-VL), and speaks the answer back in Sinhala via text-to-speech.

---

## What This Repo Contains

| File | Purpose |
|------|--------|
| **`cam_vl.py`** | Main app: webcam + Sinhala voice → STT → translate to English → VLM (image + question) → translate answer to Sinhala → TTS → play audio. |
| **`stt_si.py`** | Standalone **Sinhala Speech-to-Text** using Google Speech REST API (record with `r`, stop with `s`). |
| **`tts_si.py`** | Standalone **Sinhala Text-to-Speech** using Google TTS REST API; prompts for text and saves a WAV file. |
| **`translate_en_to_si.py`** | **English → Sinhala** translation via Google Cloud Translation API (API key or `google-cloud-translate` client). |
| **`requirements.txt`** | Python dependencies (Google Cloud, dotenv, etc.). The main pipeline also needs PyTorch, Transformers, and other libs — see below. |
| **`.env.example`** | Template for required API keys. Copy to `.env` and fill in your keys. |

---

## Features

- **Vision–language assistive flow**: Ask in Sinhala about what the camera sees; get a spoken Sinhala answer.
- **Sinhala STT**: Google Speech API (`si-LK`).
- **Sinhala TTS**: Google Text-to-Speech (`si-LK`), LINEAR16 WAV.
- **Translation**: Sinhala ↔ English via Google Translate (used to talk to the VLM in English and to respond in Sinhala).
- **VLM**: Qwen2.5-VL-3B-Instruct (4-bit quantized on GPU when available).

---

## Requirements

### 1. Python

- Python 3.8+ (3.10+ recommended for `cam_vl.py`).

### 2. API Keys (all in `.env`)

You must set these in a `.env` file (see `.env.example`):

| Variable | Purpose |
|----------|--------|
| `GOOGLE_STT_API_KEY` | Google Cloud Speech-to-Text (REST). |
| `GOOGLE_TRANSLATE_API_KEY` | Google Cloud Translation API. |
| `GOOGLE_TTS_API_KEY` | Google Cloud Text-to-Speech. |

Create the keys in [Google Cloud Console](https://console.cloud.google.com/) and enable: **Speech-to-Text API**, **Translation API**, **Text-to-Speech API**.

### 3. Hardware / System

- **Webcam**: Required for `cam_vl.py` (default device `0`).
- **Microphone**: Required for `cam_vl.py` and `stt_si.py`.
- **GPU**: Optional but recommended for `cam_vl.py` (CUDA); runs on CPU if no GPU.
- **Playback (Linux)**: `cam_vl.py` uses `aplay` to play WAV; install `alsa-utils` if needed.

### 4. Model for `cam_vl.py`

The script expects the **Qwen2.5-VL-3B-Instruct** model on disk at:

```text
models/Qwen2.5-VL-3B-Instruct/
```

Download the model (e.g. from Hugging Face) and place it there, or change `MODEL_PATH` in `cam_vl.py` to your path.

---

## Setup

### 1. Clone and enter the project

```bash
cd /path/to/object_detection_&_stt
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# or:  .venv\Scripts\activate   # Windows
```

### 3. Install dependencies

**From `requirements.txt` (Google APIs, dotenv, etc.):**

```bash
pip install -r requirements.txt
```

**Additional dependencies for `cam_vl.py`** (VLM + webcam + audio):

```bash
pip install torch torchvision
pip install transformers accelerate bitsandbytes
pip install qwen-vl-utils
pip install opencv-python Pillow numpy sounddevice requests python-dotenv
```

(Adjust for your OS and CUDA version if you use GPU.)

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env and set:
#   GOOGLE_STT_API_KEY=...
#   GOOGLE_TRANSLATE_API_KEY=...
#   GOOGLE_TTS_API_KEY=...
```

### 5. Download the VLM (for `cam_vl.py` only)

Ensure **Qwen2.5-VL-3B-Instruct** is available at `models/Qwen2.5-VL-3B-Instruct/` (or update `MODEL_PATH` in `cam_vl.py`).

---

## Usage

### Main pipeline: Vision–language with Sinhala voice (`cam_vl.py`)

1. Run:
   ```bash
   python cam_vl.py
   ```
2. A window **"Assistive Vision"** shows the webcam feed.
3. **`r`** — Start recording Sinhala speech.
4. **`r`** again — Stop recording and run the full pipeline:
   - STT (Sinhala → text)
   - Translate to English
   - Capture current frame and run VLM with (image, English question)
   - Translate English answer to Sinhala
   - TTS (Sinhala → WAV) and play (e.g. via `aplay`)
5. **`q`** — Quit.

Generated WAV is saved to `logs/output.wav`. End-to-end latency is printed after each run.

---

### Sinhala Speech-to-Text only (`stt_si.py`)

```bash
python stt_si.py
```

- **`r`** — Start recording.
- **`s`** — Stop recording and run recognition; transcript is printed.
- Requires `GOOGLE_STT_API_KEY` in `.env`.

---

### Sinhala Text-to-Speech only (`tts_si.py`)

```bash
python tts_si.py
```

- Enter Sinhala text when prompted; audio is saved to `sinhala_tts.wav` (or pass a custom path to `SinhalaTTS(output_path=...)`).
- Requires `GOOGLE_TTS_API_KEY` in `.env`.

---

### English → Sinhala translation only (`translate_en_to_si.py`)

```bash
python translate_en_to_si.py
```

- Interactive: enter English text; Sinhala translation is printed. Type `quit`, `exit`, or `q` to exit.
- Requires `GOOGLE_TRANSLATE_API_KEY` in `.env` (or Google Cloud credentials if using the client library).

---

## Project layout (relevant paths)

```text
object_detection_&_stt/
├── README.md
├── requirements.txt
├── .env.example
├── .env                 # you create this; not committed
├── .gitignore
├── cam_vl.py            # main vision–language + Sinhala pipeline
├── stt_si.py            # Sinhala STT only
├── tts_si.py            # Sinhala TTS only
├── translate_en_to_si.py
├── models/              # ignored by git; put Qwen2.5-VL-3B-Instruct here
│   └── Qwen2.5-VL-3B-Instruct/
└── logs/                # ignored by git; cam_vl.py writes output.wav here
```

---

## Troubleshooting

- **Missing API keys**: Ensure `.env` exists and all three `GOOGLE_*` keys are set. Scripts that need them will fail with a clear error if a key is missing.
- **Webcam not opening**: Check that no other app is using the camera and that `cv2.VideoCapture(0)` is correct for your system (you can change the device index in `cam_vl.py`).
- **No audio playback**: On Linux, install `aplay` (e.g. `alsa-utils`). You can still inspect `logs/output.wav` manually.
- **Out of GPU memory**: The script uses 4-bit quantization; if it still OOMs, use a smaller model or run on CPU (slower).
- **Model not found**: Ensure `models/Qwen2.5-VL-3B-Instruct/` exists and matches `MODEL_PATH` in `cam_vl.py`.

---

## License and APIs

- This project uses **Google Cloud** APIs (Speech-to-Text, Translation, Text-to-Speech). Your use is subject to Google’s terms and billing.
- The **Qwen2.5-VL** model has its own license (see the model repo on Hugging Face).

---

## Summary

| Goal | Command | Notes |
|------|--------|--------|
| Full assistive pipeline (camera + Sinhala Q&A) | `python cam_vl.py` | Needs model + all 3 API keys + webcam + mic |
| Sinhala speech → text | `python stt_si.py` | Needs `GOOGLE_STT_API_KEY` |
| Sinhala text → speech | `python tts_si.py` | Needs `GOOGLE_TTS_API_KEY` |
| English → Sinhala text | `python translate_en_to_si.py` | Needs `GOOGLE_TRANSLATE_API_KEY` |

Everything is driven by the scripts above and the environment variables in `.env`; there are no separate config files to edit for basic use.
