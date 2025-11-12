# ============================================
#  🔊 Аудіо → Google Speech-to-Text (uk-UA)
#  🖼️ Зображення → Google Vision API (OCR)
#  🧠 Аналіз задач → Vertex AI (Gemini)
# ============================================

import os
import io
import json
import subprocess
import tempfile
import traceback
from typing import Optional

from google.cloud import speech_v1 as speech
from google.cloud import vision
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig


# -----------------------------
# Налаштування Google Credentials
# -----------------------------
def _setup_google_credentials() -> str:
    """Створює тимчасовий credentials.json з GOOGLE_CREDENTIALS_JSON"""
    google_creds_json = os.environ.get("GOOGLE_CREDENTIALS_JSON")
    if not google_creds_json:
        raise ValueError("❌ GOOGLE_CREDENTIALS_JSON не знайдено у змінних середовища!")

    creds_path = "/tmp/google_credentials.json"
    with open(creds_path, "w", encoding="utf-8") as f:
        f.write(google_creds_json)

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
    return creds_path


_CREDS_PATH = _setup_google_credentials()

# -----------------------------
# Константи
# -----------------------------
SPEECH_LANGUAGE = os.getenv("SPEECH_LANGUAGE", "uk-UA")

# -----------------------------
# Конвертація аудіо → WAV 16kHz mono
# -----------------------------
def _convert_to_wav_16k_mono(input_path: str) -> str:
    """Конвертує будь-яке аудіо/відео у WAV PCM 16-bit, mono, 16000 Hz (через ffmpeg)."""
    fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-y", "-i", input_path, "-vn",
        "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", out_path,
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        if os.path.exists(out_path):
            os.remove(out_path)
        raise RuntimeError(f"ffmpeg: помилка конвертації ({e})")
    return out_path


# -----------------------------
# Google Speech-to-Text
# -----------------------------
def transcribe_audio(input_path: str) -> Optional[str]:
    wav_path = None
    try:
        wav_path = _convert_to_wav_16k_mono(input_path)
        with open(wav_path, "rb") as f:
            content = f.read()

        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code=SPEECH_LANGUAGE,
            enable_automatic_punctuation=True,
            model="latest_long",
        )

        client = speech.SpeechClient()
        response = client.recognize(config=config, audio=audio)
        if not response.results:
            return None

        text = " ".join(
            r.alternatives[0].transcript.strip()
            for r in response.results
            if r.alternatives
        )
        return text or None
    except Exception:
        return None
    finally:
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception:
                pass


# -----------------------------
# Google Vision OCR
# -----------------------------
def extract_text_from_image(image_path: str) -> Optional[str]:
    try:
        client = vision.ImageAnnotatorClient()
        with open(image_path, "rb") as img:
            content = img.read()
        image = vision.Image(content=content)
        response = client.text_detection(image=image)
        if response.error.message:
            raise RuntimeError(f"Vision API error: {response.error.message}")
        if not response.text_annotations:
            return None
        return (response.text_annotations[0].description or "").strip()
    except Exception:
        return None


# -----------------------------
# Vertex AI (Gemini) — аналіз задач
# -----------------------------
def analyze_task_with_ai(prompt: str, raw_text: str, timeout_sec: int = 20) -> Optional[str]:
    """Викликає Gemini-модель Vertex AI і повертає структуровану відповідь."""
    try:
        vertexai.init(project="task-dispatcher-bot", location="us-central1")
        model = GenerativeModel("gemini-1.5-flash")

        system_prompt = (
            prompt
            + "\n\nВідповідай українською. "
            + "Поверни лише п'ять рядків строго у формі:\n"
            + "Назва: ...\nТег: ...\nДедлайн: ...\nПріоритет: ...\nОпис: ..."
        )

        generation_config = GenerationConfig(
            temperature=0.2,
            top_p=0.9,
            max_output_tokens=512,
        )

        parts = [
            {"role": "user", "parts": [system_prompt]},
            {"role": "user", "parts": [f"Повідомлення:\n{raw_text}"]},
        ]

        # ✅ Прибрано timeout параметр — не підтримується у поточному SDK
        resp = model.generate_content(
            parts,
            generation_config=generation_config
        )

        text = getattr(resp, "text", "").strip()
        if not text:
            print("⚠️ Vertex AI не повернув текст.")
            return None

        return text

    except Exception as e:
        print("❌ Vertex AI error:", str(e))
        import traceback
        traceback.print_exc()
        return None
