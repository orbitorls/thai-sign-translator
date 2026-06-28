"""Shared PoseT5 generation defaults."""
from __future__ import annotations

from tsl.inference.pose_t5_translator import PoseT5Translator


def default_generation_config() -> dict[str, int | float]:
    return {
        "max_new_tokens": PoseT5Translator.DEFAULT_MAX_NEW_TOKENS,
        "beam_size": PoseT5Translator.DEFAULT_BEAM_SIZE,
        "no_repeat_ngram_size": PoseT5Translator.DEFAULT_NO_REPEAT_NGRAM_SIZE,
        "repetition_penalty": PoseT5Translator.DEFAULT_REPETITION_PENALTY,
        "length_penalty": PoseT5Translator.DEFAULT_LENGTH_PENALTY,
    }


__all__ = ["default_generation_config"]
