"""Language detection wrapper. Returns ISO 639-1 code or None."""

from app.logging_config import get_logger

logger = get_logger(__name__)


def detect_language(text: str) -> str | None:
    """Detect document language. Returns "en", "es", etc. None if unknown."""
    sample = text.strip()
    if len(sample) < 20:
        return None

    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0  # deterministic
        return detect(sample[:2000])
    except Exception:
        logger.debug("language_detect_failed")
        return None
