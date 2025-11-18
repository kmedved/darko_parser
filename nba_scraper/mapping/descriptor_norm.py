"""Utilities for normalizing action descriptors and extracting style flags."""
from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence, Tuple

DESC_SYNONYMS = {
    "step back": "stepback",
    "step-back": "stepback",
    "alley oop": "alleyoop",
    "alley-oop": "alleyoop",
    "finger roll": "fingerroll",
    "pull up": "pullup",
    "fade away": "fadeaway",
    "put back": "putback",
    "tip": "tipin",
    "tip in": "tipin",
}

STYLE_TOKENS = {
    "driving",
    "running",
    "pullup",
    "stepback",
    "floating",
    "fadeaway",
    "bank",
    "reverse",
    "alleyoop",
    "fingerroll",
    "tipin",
    "putback",
    "cutting",
}

_ws_re = re.compile(r"\s+")
_hyphen_re = re.compile(r"[-\s]+")


def canon_str(value: Optional[str]) -> str:
    """Canonicalize descriptor text by lowercasing and collapsing whitespace."""
    if not value:
        return ""
    lowered = value.strip().lower()
    lowered = _ws_re.sub(" ", lowered)
    lowered = _hyphen_re.sub(" ", lowered)
    return lowered.strip()


def _tokenize_descriptor(text: str) -> List[str]:
    """
    Tokenize the descriptor text, applying multi-word and single-word synonyms.
    Assumes `text` has already been normalized via `canon_str`.
    """
    if not text:
        return []

    # Multi-word synonyms on the canonical string.
    for phrase, replacement in DESC_SYNONYMS.items():
        if " " in phrase:
            phrase_norm = canon_str(phrase)
            if phrase_norm and phrase_norm in text:
                text = text.replace(phrase_norm, replacement)

    tokens: List[str] = []
    for raw_token in text.split():
        token = DESC_SYNONYMS.get(raw_token, raw_token)
        tokens.append(token)
    return tokens


def _extract_styles(tokens: Sequence[str]) -> Tuple[List[str], List[str]]:
    style_flags: List[str] = []
    remaining: List[str] = []
    for token in tokens:
        if token in STYLE_TOKENS and token not in style_flags:
            style_flags.append(token)
        else:
            remaining.append(token)
    return remaining, style_flags


def normalize_descriptor(raw: Optional[str]) -> Tuple[str, List[str]]:
    """Normalize a descriptor string into a core token and style flags.

    Parameters
    ----------
    raw:
        Raw descriptor string from the upstream feed.

    Returns
    -------
    tuple
        A pair of (core_descriptor, style_flags).
    """

    normalized = canon_str(raw)
    if not normalized:
        return "", []

    tokens = _tokenize_descriptor(normalized)
    remaining, style_flags = _extract_styles(tokens)
    descriptor_core = " ".join(remaining).strip()
    return descriptor_core, style_flags
