"""Translation layer that converts Japanese text to Korean text."""

from typing import Any, Dict, List

from deep_translator import GoogleTranslator


class JapaneseToKoreanTranslator:
    """Wraps deep-translator so we can swap implementations if needed."""

    def __init__(self, source: str = "ja", target: str = "ko"):
        self.client = GoogleTranslator(source=source, target=target)

    def translate_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Translate the text portion of each segment while preserving timings.
        """
        translated_segments: List[Dict[str, Any]] = []

        for segment in segments:
            translated = self.client.translate(segment["text"])
            translated_segments.append(
                {
                    **segment,
                    "text": translated.strip(),
                }
            )

        return translated_segments
