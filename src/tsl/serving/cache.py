"""Shared translator cache for serving runtimes."""
from __future__ import annotations

_translator_cache: dict[str, object] = {}
_active_translator: object | None = None


def clear_translator_cache() -> int:
    global _active_translator
    count = len(_translator_cache)
    _translator_cache.clear()
    _active_translator = None
    return count


def get_translator_cache() -> dict[str, object]:
    return _translator_cache


def get_active_translator_value() -> object | None:
    return _active_translator


def set_active_translator_value(translator: object | None) -> None:
    global _active_translator
    _active_translator = translator


def cache_translator(model_id: str, translator: object) -> None:
    _translator_cache[model_id] = translator
