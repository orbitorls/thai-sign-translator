import numpy as np
import pytest

from tsl.text.tokenizer import SPECIAL_TOKENS, CharTokenizer


def test_special_token_ids():
    tok = CharTokenizer()
    assert tok.pad_id == 0
    assert tok.bos_id == 1
    assert tok.eos_id == 2
    assert tok.unk_id == 3
    assert SPECIAL_TOKENS == ("<pad>", "<bos>", "<eos>", "<unk>")


def test_fit_grows_vocab():
    tok = CharTokenizer()
    tok.fit(["สวัสดี", "ครับ"])
    expected = 4 + len({"ส", "ว", "ั", "ด", "ี", "ค", "ร", "ั", "บ"})
    assert tok.vocab_size == expected
    # All special tokens still at their reserved ids.
    assert tok.pad_id == 0
    assert tok.bos_id == 1
    assert tok.eos_id == 2
    assert tok.unk_id == 3


def test_fit_idempotent():
    tok = CharTokenizer()
    tok.fit(["abc", "def"])
    size_after_first = tok.vocab_size
    tok.fit(["abc", "def"])
    assert tok.vocab_size == size_after_first
    tok.fit(["a"])
    assert tok.vocab_size == size_after_first


def test_encode_basic():
    tok = CharTokenizer(["สวัสดี"])
    ids = tok.encode("สวัสดี")
    # All ids decode back to the original text.
    assert tok.decode(ids) == "สวัสดี"
    # And every id is >= unk_id (i.e. not a special token).
    for i in ids:
        assert i >= tok.unk_id


def test_encode_unk_for_unknown():
    tok = CharTokenizer(["ab"])
    ids = tok.encode("ac")
    # 'a' is known, 'c' is not.
    assert ids[0] != tok.unk_id
    assert ids[1] == tok.unk_id


def test_encode_with_bos_eos():
    tok = CharTokenizer(["ab"])
    ids = tok.encode("ab", add_bos=True, add_eos=True)
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id
    assert tok.decode(ids) == "ab"


def test_decode_strip_special():
    tok = CharTokenizer(["ab"])
    ids = [tok.pad_id, tok.bos_id, tok.encode("ab")[0], tok.encode("ab")[1], tok.unk_id]
    assert tok.decode(ids, strip_special=True) == "ab"
    # With strip_special=False the unk shows up as the literal token.
    assert tok.decode(ids, strip_special=False).startswith("<pad><bos>")


def test_decode_stops_at_eos():
    tok = CharTokenizer(["ab"])
    ids = [tok.encode("a")[0], tok.eos_id, tok.encode("b")[0]]
    assert tok.decode(ids, strip_special=True) == "a"
    # Without strip_special, decoding continues past eos.
    decoded = tok.decode(ids, strip_special=False)
    assert "a" in decoded
    assert decoded.endswith("b")


def test_pad_to_arrays_shapes_and_padding():
    tok = CharTokenizer(["abc"])
    short = tok.encode("a")
    long = tok.encode("abc")
    padded, lengths = tok.pad_to_arrays([short, long], add_bos=True, add_eos=True)
    # short length 1 -> +bos +eos = 3. long length 3 -> +bos +eos = 5.
    assert padded.shape == (2, 5)
    assert padded.dtype == np.int64
    # Padding is pad_id.
    assert int(padded[0, 3]) == tok.pad_id
    assert int(padded[0, 4]) == tok.pad_id
    # bos/eos flank the real chars.
    assert int(padded[0, 0]) == tok.bos_id
    assert int(padded[0, 2]) == tok.eos_id
    assert int(padded[1, 0]) == tok.bos_id
    assert int(padded[1, 4]) == tok.eos_id


def test_lengths_count_real_chars():
    tok = CharTokenizer(["abc"])
    short = tok.encode("a")
    long = tok.encode("abc")
    _, lengths = tok.pad_to_arrays([short, long], add_bos=True, add_eos=True)
    assert lengths.tolist() == [1, 3]


def test_pad_to_arrays_no_specials():
    tok = CharTokenizer(["abc"])
    padded, lengths = tok.pad_to_arrays(
        [tok.encode("a"), tok.encode("abc")], add_bos=False, add_eos=False
    )
    assert padded.shape == (2, 3)
    assert int(padded[0, 1]) == tok.pad_id
    assert int(padded[0, 2]) == tok.pad_id
    assert lengths.tolist() == [1, 3]


def test_pad_to_arrays_empty_batch():
    tok = CharTokenizer()
    padded, lengths = tok.pad_to_arrays([], add_bos=True, add_eos=True)
    assert padded.shape == (0, 0)
    assert lengths.shape == (0,)


def test_encode_batch_with_max_len_truncates():
    tok = CharTokenizer(["abcde"])
    ids, lengths = tok.encode_batch(["abcde", "ab"], max_len=3)
    assert lengths == [3, 2]
    assert tok.decode(ids[0]) == "abc"


def test_encode_batch_lengths_exclude_specials():
    tok = CharTokenizer(["ab"])
    ids, lengths = tok.encode_batch(["ab"])
    assert lengths == [2]
    # pad_to_arrays (which adds bos/eos) reports the same real length.
    _, real_lengths = tok.pad_to_arrays(ids, add_bos=True, add_eos=True)
    assert real_lengths.tolist() == [2]


def test_round_trip_thai_sentence():
    tok = CharTokenizer(["ฉันกินข้าว"])
    text = "ฉันกินข้าว"
    ids = tok.encode(text)
    assert tok.decode(ids) == text
    # With bos/eos framing the round trip still works.
    assert tok.decode(tok.encode(text, add_bos=True, add_eos=True)) == text


def test_decode_unknown_id_returns_empty_string():
    tok = CharTokenizer(["a"])
    # Pick an id that is definitely not in the vocab.
    bogus_id = tok.vocab_size + 100
    assert tok.decode([bogus_id]) == ""


def test_unknown_char_at_encode_uses_unk_id():
    tok = CharTokenizer()  # no chars fitted
    ids = tok.encode("x")
    assert ids == [tok.unk_id]


def test_constructor_with_texts_fits_immediately():
    tok = CharTokenizer(["ab", "cd"])
    assert tok.vocab_size == 4 + 4
    # The first new char ('a') is right after the specials.
    assert tok.decode([4]) == "a"
    assert tok.decode([5]) == "b"
    assert tok.decode([6]) == "c"
    assert tok.decode([7]) == "d"
