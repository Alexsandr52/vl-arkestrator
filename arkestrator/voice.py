"""Robot speech output and classification of operator utterances.

Speech output goes through a ``Speaker``. ``LogSpeaker`` keeps the phrases for the UI; the
robot speaker (silero ru → ``AudioClient.PlayStream`` as in box ``say_ru.py``) is added in P3.
"""

from __future__ import annotations

import re
import threading
from collections import deque
from typing import Optional


class Speaker:
    def say(self, text: str) -> None:
        raise NotImplementedError

    def recent(self) -> list[str]:
        return []


class LogSpeaker(Speaker):
    def __init__(self, capacity: int = 50):
        self._lock = threading.Lock()
        self._said: deque[str] = deque(maxlen=capacity)

    def say(self, text: str) -> None:
        with self._lock:
            self._said.append(text)

    def recent(self) -> list[str]:
        with self._lock:
            return list(self._said)


def normalize(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^\w\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# «не отпускай», «я не держу», «нет» never release the hand.
NEGATIONS = ("не", "нет", "ни")
# Word stems; matched at word starts.
RELEASE_STEMS = ("отпуст", "отпуск", "отдай", "забрал", "забираю", "беру", "держу", "разожми")
CANCEL_STEMS = ("отмен", "стоп", "хватит", "стой", "прекрати")
FETCH_STEMS = ("принес", "найди", "найти", "возьм", "взять", "дай", "подай", "достань", "неси", "притащ", "захвати")
ATTRIBUTE_STEMS = {
    "красн": "красный",
    "син": "синий",
    "голуб": "голубой",
    "зелен": "зелёный",
    "желт": "жёлтый",
    "бел": "белый",
    "черн": "чёрный",
    "прозрачн": "прозрачный",
    "больш": "большой",
    "маленьк": "маленький",
    "пуст": "пустой",
}


def _has_stem(words: list[str], stems) -> Optional[str]:
    for word in words:
        for stem in stems:
            if word.startswith(stem):
                return stem
    return None


def classify_utterance(text: str) -> str:
    """``release`` | ``cancel`` | ``command`` | ``empty``.

    Release/cancel are short control phrases; anything else is treated as a command.
    """
    words = normalize(text).split()
    if not words:
        return "empty"
    negated = any(w in NEGATIONS or w.startswith("нельз") for w in words)
    if len(words) <= 4 and not negated and _has_stem(words, RELEASE_STEMS):
        return "release"
    if len(words) <= 3 and _has_stem(words, CANCEL_STEMS):
        return "cancel"
    return "command"


def parse_fetch_keywords(text: str, catalog: list[dict]) -> Optional[dict]:
    """Keyword fallback when the VLM is unavailable: profile by synonym stem, colour words."""
    words = normalize(text).split()
    found = None
    for entry in catalog:
        stem = _has_stem(words, entry["synonyms_ru"])
        if stem:
            found = (entry, stem)
            break
    if found is None:
        return None
    entry, _ = found
    attributes = sorted({value for stem, value in ATTRIBUTE_STEMS.items() if _has_stem(words, (stem,))})
    return {"profile": entry["name"], "object": entry["display_ru"], "attributes": attributes}
