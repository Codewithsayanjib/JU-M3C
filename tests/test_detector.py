"""Tests for the leakage audit. Run with: pytest. No data download required."""
import json
from pathlib import Path

from ju_m3c.audit import audit, jaccard, containment, shingles, text_links, DetectorConfig

FIX = Path(__file__).parent / "fixtures"


def _load(name):
    return [json.loads(l) for l in (FIX / name).read_text().splitlines() if l.strip()]


def test_shingle_similarity_identical_after_normalisation():
    a = "A SIXTY-TWO year old man; presented with dyspnoea."
    b = "a sixty two year old man presented with dyspnoea"
    assert jaccard(shingles(a), shingles(b)) == 1.0


def test_containment_of_shorter_in_longer():
    long = "the quick brown fox jumps over the lazy dog near the river bank at dawn each morning"
    short = "the quick brown fox jumps over the lazy dog near the river bank at dawn"
    assert containment(shingles(short), shingles(long)) == 1.0


def test_audit_finds_both_layers_on_toy_data():
    train, test = _load("toy_train.jsonl"), _load("toy_test.jsonl")
    r = audit(train, test)
    assert r["n_test"] == 3
    assert r["leaked"] == 2
    assert r["article_layer_only"] == 1          # tst-art: shared pmcid only
    assert r["text_layer_only"] == 1             # tst-dup: reformatted text only
    assert set(r["leaked_ids"]) == {"tst-art", "tst-dup"}
    assert "tst-clean" not in r["leaked_ids"]


def test_audit_clean_when_no_overlap():
    train = [{"id": "t1", "text": "an isolated cardiology case with no counterpart", "pmcid": "A"}]
    test = [{"id": "x1", "text": "a totally unrelated dermatology presentation", "pmcid": "B"}]
    r = audit(train, test)
    assert r["leaked"] == 0


def test_text_links_within_one_set():
    docs = {
        "a": "a sixty two year old man presented with progressive dyspnoea and oedema over three weeks",
        "b": "A SIXTY-TWO year old man presented with progressive dyspnoea and oedema over three weeks!",
        "c": "a young woman with an acute migraine responding well to triptan therapy at home",
    }
    links = text_links(docs, DetectorConfig(min_words=5))
    assert ("a", "b") in links
    assert not any("c" in pair for pair in links)
