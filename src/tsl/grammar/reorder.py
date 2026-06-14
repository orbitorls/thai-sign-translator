"""(Stretch) Rule-based reorder of recognized sign glosses into a Thai sentence."""
from __future__ import annotations

_LEXICON: dict[str, tuple[str, str]] = {
    "i": ("subject", "ฉัน"),
    "you": ("subject", "คุณ"),
    "drink": ("verb", "ดื่ม"),
    "eat": ("verb", "กิน"),
    "go": ("verb", "ไป"),
    "water": ("object", "น้ำ"),
    "rice": ("object", "ข้าว"),
}

_SVO_ORDER = ("subject", "verb", "object")


def reorder_to_thai(words: list[str]) -> str:
    slots: dict[str, list[str]] = {"subject": [], "verb": [], "object": []}
    for raw in words:
        key = raw.strip().lower()
        if key in _LEXICON:
            role, thai = _LEXICON[key]
            slots[role].append(thai)
        else:
            slots["object"].append(raw)
    ordered: list[str] = []
    for role in _SVO_ORDER:
        ordered.extend(slots[role])
    return " ".join(ordered)
