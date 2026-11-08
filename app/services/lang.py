"""Language detection wrapper. Returns ISO 639-1 code or None."""

import re
import threading
from collections.abc import Iterable
from typing import Literal

from app.logging_config import get_logger

logger = get_logger(__name__)

# Ingestion calls this from worker threads. langdetect loads its language
# profiles into a module-level factory on first use, without a lock, so two
# threads arriving together can both load them, or one can read them half
# loaded.
_lock = threading.Lock()


def detect_language(text: str) -> str | None:
    """Detect document language. Returns "en", "es", etc. None if unknown."""
    sample = text.strip()
    if len(sample) < 20:
        return None

    try:
        from langdetect import DetectorFactory, detect

        with _lock:
            DetectorFactory.seed = 0  # deterministic
            return detect(sample[:2000])
    except Exception:
        logger.debug("language_detect_failed")
        return None


ReplyLanguage = Literal["en", "es"]

# Chat messages are often shorter than what langdetect handles, so marks and
# common words found in one language only decide first.
_SPANISH_MARKS = re.compile(r"[¿¡ñáéíóú]", re.IGNORECASE)
_WORD = re.compile(r"[a-z]+")
_SPANISH_WORDS = frozenset(
    "el la los las del de que como cual cuales donde cuando por para con sin un una "
    "unos unas es son hay y en se mi tu su sobre esto eso hola gracias puedo".split()
)
_ENGLISH_WORDS = frozenset(
    "the of and to in is are what how which where when why who does do can for with "
    "without about an my your this that it hello hi thanks".split()
)


def _guess_reply_language(text: str) -> ReplyLanguage | None:
    if _SPANISH_MARKS.search(text):
        return "es"
    words = _WORD.findall(text.lower())
    spanish = sum(w in _SPANISH_WORDS for w in words)
    english = sum(w in _ENGLISH_WORDS for w in words)
    if spanish != english:
        return "es" if spanish > english else "en"
    detected = detect_language(text)
    if detected == "es":
        return "es"
    if detected == "en":
        return "en"
    return None


def reply_language(user_messages: Iterable[str]) -> ReplyLanguage:
    """Spanish or English, for an answer written without the model (audit N8).

    Takes the user's messages newest first; the first one that settles it
    wins. English, the language of the corpus, when none does.
    """
    for text in user_messages:
        language = _guess_reply_language(text)
        if language is not None:
            return language
    return "en"
