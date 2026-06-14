"""EAF (ELAN Annotation Format) parser for the NSTDA Thai Sign Language
multi-tier release.

The NSTDA release ships one ``.eaf`` file per video. Each file uses
the standard ELAN XML schema:

- ``<TIME_ORDER>`` contains ``<TIME_SLOT TIME_SLOT_ID="tsN" TIME_VALUE="ms"/>``
  entries that map slot ids to millisecond timestamps.
- ``<TIER TIER_ID="...">`` blocks contain ``<ANNOTATION>`` -> ``<ALIGNABLE_ANNOTATION
  TIME_SLOT_REF1="tsA" TIME_SLOT_REF2="tsB">`` -> ``<ANNOTATION_VALUE>`` chains.

For Thai SLT we only care about the tiers that carry text or gloss
supervision: ``CC``, ``Gloss``, ``Gloss Labeling``, ``Left-hand`` and
``Right-hand``. Every other tier is silently ignored. The result is a
flat, time-sorted list of :class:`EafAnnotation` records that downstream
code can iterate to build sentence-level supervision targets.

This module is stdlib-only (no lxml, no pandas) so it can be imported
from anywhere in the pipeline, including lightweight dataset adapters.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass

__all__ = ["EafAnnotation", "EafDocument", "parse_eaf"]

# Tiers we keep. Anything else is dropped without a warning because the
# NSTDA release occasionally adds extra metadata tiers per release.
_KNOWN_TIERS: frozenset[str] = frozenset(
    {"CC", "Gloss", "Gloss Labeling", "Left-hand", "Right-hand"}
)

# Local element / attribute names we look at. We match by local name so
# the parser works for both namespaced (real ELAN files use a default
# xmlns) and non-namespaced XML serialisations.
_TAG_TIME_SLOT = "TIME_SLOT"
_TAG_TIER = "TIER"
_TAG_ANNOTATION = "ANNOTATION"
_TAG_ALIGNABLE_ANNOTATION = "ALIGNABLE_ANNOTATION"
_TAG_ANNOTATION_VALUE = "ANNOTATION_VALUE"

_ATTR_TIME_SLOT_ID = "TIME_SLOT_ID"
_ATTR_TIME_VALUE = "TIME_VALUE"
_ATTR_TIER_ID = "TIER_ID"
_ATTR_REF1 = "TIME_SLOT_REF1"
_ATTR_REF2 = "TIME_SLOT_REF2"


@dataclass(frozen=True)
class EafAnnotation:
    """One annotation in an EAF file that belongs to a known tier."""

    tier: str
    start_ms: int
    end_ms: int
    value: str


@dataclass(frozen=True)
class EafDocument:
    """Parsed view of a single EAF file."""

    path: str
    annotations: list[EafAnnotation]


def _local_name(elem: ET.Element) -> str:
    """Return the local element name, stripping any XML namespace."""
    tag = elem.tag
    if isinstance(tag, str) and tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def parse_eaf(path: str) -> EafDocument:
    """Parse an EAF file and return only the Thai-SLT-relevant tiers.

    Parameters
    ----------
    path:
        Filesystem path to a ``.eaf`` (ELAN XML) file.

    Returns
    -------
    EafDocument
        ``annotations`` is sorted by ``(start_ms, tier)`` so iteration
        order is deterministic regardless of the input file.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist or is not a regular file.
    xml.etree.ElementTree.ParseError
        If the file is not well-formed XML.
    ValueError
        If an ``<ALIGNABLE_ANNOTATION>`` references a ``TIME_SLOT`` that
        is not defined in ``<TIME_ORDER>``, or omits the required slot
        references.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    tree = ET.parse(path)
    root = tree.getroot()

    time_slots: dict[str, int] = {}
    for elem in root.iter():
        if _local_name(elem) != _TAG_TIME_SLOT:
            continue
        slot_id = elem.get(_ATTR_TIME_SLOT_ID)
        slot_value = elem.get(_ATTR_TIME_VALUE)
        if slot_id is None or slot_value is None:
            continue
        time_slots[slot_id] = int(slot_value)

    annotations: list[EafAnnotation] = []
    for tier in root.iter():
        if _local_name(tier) != _TAG_TIER:
            continue
        tier_id = tier.get(_ATTR_TIER_ID)
        if tier_id is None or tier_id not in _KNOWN_TIERS:
            continue
        for ann in tier:
            if _local_name(ann) != _TAG_ANNOTATION:
                continue
            for alignable in ann:
                if _local_name(alignable) != _TAG_ALIGNABLE_ANNOTATION:
                    continue
                ref1 = alignable.get(_ATTR_REF1)
                ref2 = alignable.get(_ATTR_REF2)
                if ref1 is None or ref2 is None:
                    raise ValueError(
                        f"tier {tier_id!r}: ALIGNABLE_ANNOTATION missing "
                        f"TIME_SLOT_REF1/TIME_SLOT_REF2"
                    )
                if ref1 not in time_slots:
                    raise ValueError(
                        f"tier {tier_id!r}: unknown TIME_SLOT_REF1 {ref1!r}"
                    )
                if ref2 not in time_slots:
                    raise ValueError(
                        f"tier {tier_id!r}: unknown TIME_SLOT_REF2 {ref2!r}"
                    )

                value = ""
                for child in alignable:
                    if _local_name(child) == _TAG_ANNOTATION_VALUE:
                        value = child.text or ""
                        break

                annotations.append(
                    EafAnnotation(
                        tier=tier_id,
                        start_ms=time_slots[ref1],
                        end_ms=time_slots[ref2],
                        value=value,
                    )
                )

    annotations.sort(key=lambda a: (a.start_ms, a.tier))
    return EafDocument(path=path, annotations=annotations)
