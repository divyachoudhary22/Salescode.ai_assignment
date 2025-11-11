# interrupt_filter.py
from __future__ import annotations
import asyncio
import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Dict, Set, Callable

def _env_list(key: str, default: Iterable[str]) -> List[str]:
    raw = os.getenv(key)
    if not raw:
        return list(default)
    return [w.strip() for w in raw.split(",") if w.strip()]

DEFAULT_FILLERS_EN = [
    "uh", "umm", "um", "hmm", "huh", "erm", "er", "uhh", "mm", "mmm", "ah",
    "uh-huh", "mhm", "hm", "yeahh", "hmm yeah"
]
DEFAULT_FILLERS_HI = [
    "haan", "hmm", "hmmm", "achha", "theek", "huh", "hmm haan"
]

DEFAULT_INTENTS = {
    "stop", "wait", "hold on", "hold-on", "pause", "one second", "one sec",
    "no not that", "not that", "cancel", "enough", "okay stop", "ok stop",
    "bas", "ruk", "ruko", "ruk jao", "band karo", "thodi der"
}

_WORD = re.compile(r"[0-9A-Za-z\u0900-\u097F']+")
DEVANAGARI = re.compile(r"[\u0900-\u097F]")

def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD.findall(text or "")]

def contains_devanagari(text: str) -> bool:
    return bool(DEVANAGARI.search(text or ""))

def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())

@dataclass
class ASRSegment:
    text: str
    confidence: float
    is_final: bool

@dataclass
class FilterDecision:
    should_interrupt: bool
    should_consume: bool
    reason: str

class InterruptFilter:
    """
    Real-time filter for filler vs. genuine user interruptions.
    """
    def __init__(
        self,
        ignored_words: Optional[Iterable[str]] = None,
        command_intents: Optional[Iterable[str]] = None,
        confidence_threshold: float = 0.55,
        min_chars: int = 1,
        debounce_ms: int = 180,
        logger: Optional[Callable[[str, Dict], None]] = None
    ):
        self._lock = asyncio.Lock()
        self._is_speaking = False
        self._ignored: Set[str] = set(ignored_words) if ignored_words else (
            set(_env_list("IGNORED_WORDS",
                          DEFAULT_FILLERS_EN + DEFAULT_FILLERS_HI))
        )
        self._ignored = {normalize_space(w) for w in self._ignored if w}

        self._intents: Set[str] = set(command_intents) if command_intents else (
            set(_env_list("INTERRUPT_INTENTS", DEFAULT_INTENTS))
        )
        self._intents = {normalize_space(w) for w in self._intents if w}

        self._conf_th = confidence_threshold
        self._min_chars = min_chars
        self._debounce_ms = debounce_ms
        self._aggregate_buf: List[ASRSegment] = []
        self._logger = logger or (lambda msg, kv: None)

    async def set_tts_speaking(self, speaking: bool) -> None:
        async with self._lock:
            self._is_speaking = speaking
            self._aggregate_buf.clear()

    async def decide(self, seg: ASRSegment) -> FilterDecision:
        async with self._lock:
            text = normalize_space(seg.text)
            if len(text) < self._min_chars:
                return self._emit(False, True if self._is_speaking else False,
                                  "empty_or_too_short", seg)

            if self._is_speaking is False:
                if self._has_intent(text):
                    self._log("intent_while_quiet", seg, extra={"text": text})
                    return self._emit(True, False, "intent_while_quiet", seg)
                return self._emit(False, False, "agent_quiet_passthrough", seg)

            if seg.confidence < self._conf_th:
                self._log("low_conf_ignored", seg, extra={"text": text})
                return self._emit(False, True, "low_confidence", seg)

            if self._has_intent(text):
                self._log("INTERRUPT", seg, extra={"text": text})
                return self._emit(True, False, "command_intent", seg)

            if self._is_filler_only(text):
                self._log("filler_consumed", seg, extra={"text": text})
                return self._emit(False, True, "filler_only", seg)

            self._log("meaningful_overlap_interrupt", seg, extra={"text": text})
            return self._emit(True, False, "meaningful_overlap", seg)

    async def update_ignored_words(self, new_words: Iterable[str]) -> None:
        async with self._lock:
            self._ignored = {normalize_space(w) for w in new_words if w}

    async def update_intents(self, new_intents: Iterable[str]) -> None:
        async with self._lock:
            self._intents = {normalize_space(w) for w in new_intents if w}

    def _has_intent(self, text: str) -> bool:
        t = normalize_space(text)
        for p in self._intents:
            if p in t:
                return True
        toks = tokenize(t)
        return any(tok in self._intents for tok in toks)

    def _is_filler_only(self, text: str) -> bool:
        t = normalize_space(text)
        if t in self._ignored:
            return True
        toks = tokenize(t)
        return len(toks) > 0 and all(tok in self._ignored for tok in toks)

    def _emit(self, interrupt: bool, consume: bool, reason: str, seg: ASRSegment) -> FilterDecision:
        return FilterDecision(should_interrupt=interrupt, should_consume=consume, reason=reason)

    def _log(self, tag: str, seg: ASRSegment, extra: Optional[Dict]=None) -> None:
        payload = {"tag": tag, "confidence": seg.confidence, "final": seg.is_final}
        if extra:
            payload.update(extra)
        self._logger("interrupt_filter", payload)
