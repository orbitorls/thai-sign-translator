import pytest
from tsl.grammar.reorder import reorder_to_thai


def test_osv_demo_sentence_maps_to_thai_svo():
    assert reorder_to_thai(["water", "i", "drink"]) == "ฉัน ดื่ม น้ำ"


def test_second_demo_sentence():
    assert reorder_to_thai(["rice", "you", "eat"]) == "คุณ กิน ข้าว"


def test_case_insensitive_input():
    assert reorder_to_thai(["WATER", "I", "DRINK"]) == "ฉัน ดื่ม น้ำ"


def test_subject_verb_only_no_object():
    assert reorder_to_thai(["i", "go"]) == "ฉัน ไป"


def test_unknown_word_passes_through_verbatim():
    assert reorder_to_thai(["i", "drink", "cola"]) == "ฉัน ดื่ม cola"


def test_empty_returns_empty_string():
    assert reorder_to_thai([]) == ""
