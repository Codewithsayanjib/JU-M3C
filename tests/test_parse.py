"""Tests for the benchmark output parsers. Run with: pytest."""
from ju_m3c.benchmark.parse import parse_label, strip_thinking


def test_strip_thinking_removes_block():
    out, stripped = strip_thinking("<think>reasoning here</think>Imaging report")
    assert out == "Imaging report"
    assert stripped is True


def test_strip_thinking_noop():
    out, stripped = strip_thinking("Progress note")
    assert out == "Progress note"
    assert stripped is False


def test_parse_label_strict():
    r = parse_label("Imaging report", ["Imaging report", "Progress note"])
    assert r["label"] == "Imaging report"
    assert r["strict"] is True


def test_parse_label_lenient_substring():
    r = parse_label("The answer is Progress note.", ["Imaging report", "Progress note"])
    assert r["label"] == "Progress note"
    assert r["strict"] is False
    assert r["lenient"] is True


def test_parse_label_unparseable():
    r = parse_label("I am not sure", ["Imaging report", "Progress note"])
    assert r["label"] is None
    assert r["lenient"] is False
