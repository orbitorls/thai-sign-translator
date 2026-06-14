import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from tsl.data.eaf import EafAnnotation, EafDocument, parse_eaf


_BASE_TIME_SLOTS = (
    '<TIME_SLOT TIME_SLOT_ID="ts1" TIME_VALUE="0"/>'
    '<TIME_SLOT TIME_SLOT_ID="ts2" TIME_VALUE="1000"/>'
    '<TIME_SLOT TIME_SLOT_ID="ts3" TIME_VALUE="2000"/>'
)

# Four known tiers (CC, Gloss, Left-hand, Right-hand) plus one
# tier name that must be silently ignored.
_KNOWN_AND_IGNORED_TIERS = """
<TIER TIER_ID="CC">
  <ANNOTATION>
    <ALIGNABLE_ANNOTATION TIME_SLOT_REF1="ts1" TIME_SLOT_REF2="ts2">
      <ANNOTATION_VALUE>สวัสดี</ANNOTATION_VALUE>
    </ALIGNABLE_ANNOTATION>
  </ANNOTATION>
</TIER>
<TIER TIER_ID="Gloss">
  <ANNOTATION>
    <ALIGNABLE_ANNOTATION TIME_SLOT_REF1="ts2" TIME_SLOT_REF2="ts3">
      <ANNOTATION_VALUE>สวัสดี</ANNOTATION_VALUE>
    </ALIGNABLE_ANNOTATION>
  </ANNOTATION>
</TIER>
<TIER TIER_ID="Left-hand">
  <ANNOTATION>
    <ALIGNABLE_ANNOTATION TIME_SLOT_REF1="ts1" TIME_SLOT_REF2="ts3">
      <ANNOTATION_VALUE>e15</ANNOTATION_VALUE>
    </ALIGNABLE_ANNOTATION>
  </ANNOTATION>
</TIER>
<TIER TIER_ID="Right-hand">
  <ANNOTATION>
    <ALIGNABLE_ANNOTATION TIME_SLOT_REF1="ts2" TIME_SLOT_REF2="ts3">
      <ANNOTATION_VALUE>b15</ANNOTATION_VALUE>
    </ALIGNABLE_ANNOTATION>
  </ANNOTATION>
</TIER>
<TIER TIER_ID="Ignored">
  <ANNOTATION>
    <ALIGNABLE_ANNOTATION TIME_SLOT_REF1="ts1" TIME_SLOT_REF2="ts2">
      <ANNOTATION_VALUE>should_not_appear</ANNOTATION_VALUE>
    </ALIGNABLE_ANNOTATION>
  </ANNOTATION>
</TIER>
"""


def _wrap(time_slots_xml: str, tiers_xml: str = _KNOWN_AND_IGNORED_TIERS) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<ANNOTATION_DOCUMENT>"
        "<TIME_ORDER>"
        + time_slots_xml
        + "</TIME_ORDER>"
        "<TIER_LIST>"
        + tiers_xml
        + "</TIER_LIST>"
        "</ANNOTATION_DOCUMENT>"
    )


def _write_eaf(tmp_path: Path, content: str, name: str = "sample.eaf") -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def test_parse_eaf_returns_only_known_tiers(tmp_path):
    eaf_path = _write_eaf(tmp_path, _wrap(_BASE_TIME_SLOTS))

    doc = parse_eaf(str(eaf_path))

    assert isinstance(doc, EafDocument)
    assert doc.path == str(eaf_path)
    assert len(doc.annotations) == 4
    tiers = {a.tier for a in doc.annotations}
    assert "Ignored" not in tiers
    assert tiers == {"CC", "Gloss", "Left-hand", "Right-hand"}
    for ann in doc.annotations:
        assert isinstance(ann, EafAnnotation)


def test_annotations_sorted_by_start(tmp_path):
    eaf_path = _write_eaf(tmp_path, _wrap(_BASE_TIME_SLOTS))

    doc = parse_eaf(str(eaf_path))

    starts = [a.start_ms for a in doc.annotations]
    assert starts == sorted(starts)
    # Ties on start_ms are broken by tier name (stable secondary key).
    for a, b in zip(doc.annotations, doc.annotations[1:]):
        if a.start_ms == b.start_ms:
            assert a.tier <= b.tier


def test_annotation_values_and_timestamps(tmp_path):
    eaf_path = _write_eaf(tmp_path, _wrap(_BASE_TIME_SLOTS))

    doc = parse_eaf(str(eaf_path))
    by_tier = {a.tier: a for a in doc.annotations}

    assert by_tier["CC"].value == "สวัสดี"
    assert by_tier["CC"].start_ms == 0
    assert by_tier["CC"].end_ms == 1000

    assert by_tier["Gloss"].value == "สวัสดี"
    assert by_tier["Gloss"].start_ms == 1000
    assert by_tier["Gloss"].end_ms == 2000

    assert by_tier["Left-hand"].value == "e15"
    assert by_tier["Left-hand"].start_ms == 0
    assert by_tier["Left-hand"].end_ms == 2000

    assert by_tier["Right-hand"].value == "b15"
    assert by_tier["Right-hand"].start_ms == 1000
    assert by_tier["Right-hand"].end_ms == 2000


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_eaf(str(tmp_path / "does_not_exist.eaf"))


def test_missing_time_slot_ref_raises(tmp_path):
    tiers_xml = """
<TIER TIER_ID="CC">
  <ANNOTATION>
    <ALIGNABLE_ANNOTATION TIME_SLOT_REF1="ts1" TIME_SLOT_REF2="ts999">
      <ANNOTATION_VALUE>oops</ANNOTATION_VALUE>
    </ALIGNABLE_ANNOTATION>
  </ANNOTATION>
</TIER>
"""
    eaf_path = _write_eaf(tmp_path, _wrap(_BASE_TIME_SLOTS, tiers_xml))

    with pytest.raises(ValueError, match="ts999"):
        parse_eaf(str(eaf_path))


def test_malformed_xml_raises_parse_error(tmp_path):
    eaf_path = _write_eaf(tmp_path, "<root><unclosed>", name="bad.eaf")

    with pytest.raises(ET.ParseError):
        parse_eaf(str(eaf_path))
