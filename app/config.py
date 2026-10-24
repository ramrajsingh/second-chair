import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

NEO4J_URI = os.environ.get("NEO4J_URI", "")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE") or "neo4j"

# When set (e.g. on a public deployment), every /api call needs this passcode in the X-Passcode header.
# Leave empty for local use.
APP_PASSCODE = os.environ.get("APP_PASSCODE", "")

CRUSTDATA_API_KEY = os.environ.get("CRUSTDATA_API_KEY", "")
CRUSTDATA_BASE_URL = "https://api.crustdata.com"
CRUSTDATA_API_VERSION = "2025-11-01"

SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "")
SARVAM_STT_MODEL = os.environ.get("SARVAM_STT_MODEL", "saaras:v4")
SARVAM_TTS_MODEL = os.environ.get("SARVAM_TTS_MODEL", "bulbul:v4-flash")
SARVAM_TTS_SPEAKER = os.environ.get("SARVAM_TTS_SPEAKER", "priya")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1")
# Claim spotting runs on every utterance, so it uses a faster model.
OPENAI_FAST_MODEL = os.environ.get("OPENAI_FAST_MODEL", "gpt-4.1-mini")
# Whispers are one templated sentence, so the smallest model is enough.
OPENAI_WHISPER_MODEL = os.environ.get("OPENAI_WHISPER_MODEL", "gpt-4.1-mini")

# Languages Bulbul can speak (used by app/voice.py).
TTS_LANGUAGES = {"en-IN", "hi-IN", "bn-IN", "ta-IN", "te-IN", "kn-IN", "ml-IN", "mr-IN", "gu-IN", "pa-IN", "od-IN"}
