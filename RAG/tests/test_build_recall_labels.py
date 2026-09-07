"""
Tests for eval/build_recall_labels.py's text-normalization logic.

What these tests prove:
    - _normalize collapses curly quotes to straight quotes, strips
      trademark symbols the SEC HTML inserts mid-sentence, and collapses
      whitespace -- the three real mismatches found while building
      eval/recall_labels.json against the actual corpus (see module
      docstring). Without this, a verbatim golden quote fails to match
      its own source chunk's stored text.
"""

from eval.build_recall_labels import _normalize


def test_normalize_collapses_curly_quotes():
    assert _normalize("Sony’s PlayStation") == "sony's playstation"


def test_normalize_strips_trademark_symbols():
    assert _normalize("PlayStation ® 5") == "playstation 5"
    assert _normalize("Xbox Series S™ and X™") == "xbox series s and x"


def test_normalize_collapses_whitespace_and_lowercases():
    assert _normalize("  AMD   competes\nwith  Intel. ") == "amd competes with intel."


def test_normalize_matches_across_all_differences_at_once():
    golden = "Sony PlayStation 5, the Microsoft Xbox Series S and X game consoles"
    stored = "Sony PlayStation ® 5, the Microsoft ® Xbox ® Series S™ and X™ game consoles"
    assert _normalize(golden) in _normalize(stored)
