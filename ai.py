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
from typing import Optional

from google.cloud import speech_v1 as speech
from google.cloud import vision
from googleapiclient.discovery import build as gbuild  # не обов'язково, але хай буде
# Vertex AI
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

# -----------------------------
# Налаштування Google Credentials
# -----------------------------
def _setup_google_credentials() -> str:
    """
    Налаштовує GOOGLE_APPLICATION_CREDENTIALS на основі
    змінної середовища GOOGLE_CREDENTIALS_JSON.
    Повертає шлях до файлу credentials.
    """
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
# Speech/Vision налаштування
# -----------------------------
SPEECH_LANGUAGE = os.getenv("SPEECH_LANGUAGE", "uk-UA")

# -----------------------------
# Допоміжне: конвертація → WAV
# -----------------------------
def _convert_to_wav_16k_mono(input_path: str) -> str:
    """
    Конвертує будь-яке аудіо/відео в WAV PCM 16-bit, mono, 16000 Hz.
    Вимагає наявності ffmpeg.
    """
    fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error",
        "-y",
        "-i", input_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-sample_fmt", "s16",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        try:
            os.remove(out_path)
        except Exception:
            pass
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

        parts = []
        for result in response.results:
            if result.alternatives:
                parts.append(result.alternatives[0].transcript)
        text = " ".join(t.strip() for t in parts if t and t.strip())
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
        with open(image_path, "rb") as image_file:
            content = image_file.read()
        image = vision.Image(content=content)
        response = client.text_detection(image=image)
        if response.error.message:
            raise RuntimeError(f"Vision API error: {response.error.message}")
        if not response.text_annotations:
            return None
        full_text = (response.text_annotations[0].description or "").strip()
        return full_text or None
    except Exception:
        return None

# -----------------------------
# Vertex AI (Gemini) — аналіз задач
# -----------------------------
def _init_vertex() -> tuple[str, str]:
    """
    Ініціалізує Vertex AI.
    Повертає (project_id, location).
    """
    project_id = None
    try:
        with open(_CREDS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            project_id = data.get("project_id")
    except Exception:
        pass

    if not project_id:
        # запасний варіант: можна задати через env
        project_id = os.getenv("GOOGLE_PROJECT_ID")

    location = os.getenv("VERTEX_LOCATION", "us-central1")
    if not project_id:
        raise ValueError("Не вдалося визначити project_id для Vertex AI. Додай GOOGLE_PROJECT_ID або project_id у ключі.")

    vertexai.init(project=project_id, location=location)
    return project_id, location

def analyze_task_with_ai(prompt: str, raw_text: str, timeout_sec: int = 20) -> Optional[str]:
    """
    Приймає системний промт + сирий текст чорнетки.
    Повертає структурований текст у форматі:
      Назва: ...
      Тег: ...
      Дедлайн: ...
      Пріоритет: ...
      Опис: ...
    або None, якщо щось пішло не так.
    """
    try:
        _init_vertex()
        model_name = os.getenv("VERTEX_MODEL", "gemini-1.5-flash")
        model = GenerativeModel(model_name)

        # Просимо строгий формат і українську мову
        system = (
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

        # Контент: інструкція + вихідний текст
        parts = [
            {"role": "user", "parts": [system]},
            {"role": "user", "parts": [f"Повідомлення:\n{raw_text}"]},
        ]

        resp = model.generate_content(
            parts,
            generation_config=generation_config,
            timeout=timeout_sec,
        )
        if not resp or not getattr(resp, "text", None):
            return None

        answer = (resp.text or "").strip()
        if not answer:
            return None
        return answer
    except Exception:
        return None
