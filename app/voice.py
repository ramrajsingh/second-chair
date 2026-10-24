"""Sarvam speech and translation: hears both call channels (any Indian language) and speaks whispers.

    caller audio --Saaras STT--> original text + language --Sarvam Translate--> English (agent + memory)
    agent English reply --Sarvam Translate--> caller's language --Bulbul TTS--> audio
"""

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from sarvamai import SarvamAI

from app import config

# Bulbul v4-flash persona voices, one per language. Languages without a v4 persona
# fall back to a bulbul:v3 speaker, which covers every supported language.
V4_VOICES = {
    "en-IN": "simran_en_customer",
    "hi-IN": "ritu_hi_customer",
    "bn-IN": "roopa_bn_conversational",
    "ta-IN": "vijay_ta_narration",
    "te-IN": "kavitha_te_conversation",
    "kn-IN": "chaitra_kn_conversation",
    "mr-IN": "ishita_mr_conversational",
    "gu-IN": "pooja_gu_customer",
    "pa-IN": "anand_pa_customer",
}
V3_FALLBACK_SPEAKER = "priya"

# Voices for reading typed caller lines aloud, so a typed demo still sounds like a two-voice call.
# The Hindi-English personas cover hi-IN and en-IN; bulbul:v3 speakers cover the other languages.
CALLER_VOICES = {
    "male": ("bulbul:v4-flash", "sunny_enhi_customer", "rahul"),
    "female": ("bulbul:v4-flash", "ishita_enhi_customer", "neha"),
}

# Terms speech-to-text should keep intact. Sessions add the founder, company and investor names.
KEYTERMS = ["Series A", "Series B", "seed round", "ARR", "headcount", "CTO", "CEO", "VP"]


TTS_POOL = ThreadPoolExecutor(max_workers=8)


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?।])\s+", text.strip()) if s]


@dataclass
class Heard:
    original: str
    language: str
    english: str


class Voice:
    def __init__(self, api_key: str = config.SARVAM_API_KEY):
        self.client = SarvamAI(api_subscription_key=api_key)

    def transcribe(self, audio: bytes, filename: str = "turn.webm", codec: str = "webm",
                   keyterms: list[str] | None = None) -> Heard:
        terms = (KEYTERMS + (keyterms or []))[:50]
        extra = {"keyterms": terms} if config.SARVAM_STT_MODEL == "saaras:v4" else {}
        stt = self.client.speech_to_text.transcribe(
            file=(filename, audio),
            model=config.SARVAM_STT_MODEL,
            mode="transcribe",          # keep the caller's own words; we translate separately
            language_code="unknown",    # auto-detect
            input_audio_codec=codec,
            **extra,
        )
        lang = stt.language_code or "en-IN"
        return Heard(original=stt.transcript, language=lang, english=self.to_english(stt.transcript, lang))

    def from_text(self, text: str) -> Heard:
        """Typed input path: detect the language with Sarvam Translate's auto mode."""
        if not text.strip():
            return Heard(text, "en-IN", text)
        r = self.client.text.translate(input=text, source_language_code="auto", target_language_code="en-IN",
                                       mode="code-mixed")
        lang = r.source_language_code or "en-IN"
        english = text if lang == "en-IN" else r.translated_text
        return Heard(original=text, language=lang, english=english)

    def to_english(self, text: str, lang: str) -> str:
        if not text.strip() or lang == "en-IN":
            return text
        return self.client.text.translate(input=text, source_language_code=lang,
                                          target_language_code="en-IN").translated_text

    def from_english(self, text: str, lang: str) -> str:
        if not text.strip() or lang == "en-IN":
            return text
        return self.client.text.translate(input=text, source_language_code="en-IN", target_language_code=lang,
                                          mode="classic-colloquial", speaker_gender="Female").translated_text

    def speak_caller(self, text: str, lang: str, gender: str = "male") -> list[str]:
        v4_model, v4_speaker, v3_speaker = CALLER_VOICES.get(gender, CALLER_VOICES["male"])
        if lang in ("hi-IN", "en-IN"):
            return self.speak(text, lang, (v4_model, v4_speaker))[0]
        return self.speak(text, lang, ("bulbul:v3", v3_speaker))[0]

    def speak(self, text: str, lang: str, voice: tuple[str, str] | None = None) -> tuple[list[str], str]:
        """Return (base64 mp3 chunk per sentence, language actually spoken).

        Sentences are synthesized in parallel: about 1.6 s for a three-sentence reply instead of ~3 s.
        """
        spoken_lang = lang if lang in config.TTS_LANGUAGES else "en-IN"
        if voice:
            model, speaker = voice
        elif config.SARVAM_TTS_MODEL == "bulbul:v4-flash" and spoken_lang in V4_VOICES:
            model, speaker = "bulbul:v4-flash", V4_VOICES[spoken_lang]
        else:
            model, speaker = "bulbul:v3", config.SARVAM_TTS_SPEAKER or V3_FALLBACK_SPEAKER

        def one(sentence: str) -> str:
            r = self.client.text_to_speech.convert(
                text=sentence[:2000], language_code=spoken_lang, model=model, speaker=speaker,
                output_audio_codec="mp3", pace=1.05,
            )
            return "".join(r.audios)

        return list(TTS_POOL.map(one, sentences(text) or [text])), spoken_lang
