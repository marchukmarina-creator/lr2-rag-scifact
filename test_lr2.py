import json

import lr2_eval as ev
from lr2_core import parse_response


def test_known_retrieval_example():
    assert ev.known_metric_example() == {
        "hit3": 1.0, "recall3": 0.5, "rr3": 1 / 3,
        "hit1": 0.0, "recall1": 0.0, "rr1": 0.0,
    }


def test_parser_rejects_unknown_label_and_extra_field():
    assert not parse_response('{"label":"MAYBE","doc_id":null,"sentences":[]}', "rag", [])[0]
    assert not parse_response('{"label":"NOINFO","doc_id":null,"sentences":[],"x":1}', "rag", [])[0]


def test_parser_rejects_untransmitted_citation():
    transmitted = [{"doc_id": "42", "sentences": [{"number": 3, "text": "evidence"}]}]
    raw = json.dumps({"label": "SUPPORT", "doc_id": "42", "sentences": [4]})
    assert not parse_response(raw, "rag", transmitted)[0]


def test_parser_accepts_transmitted_citation():
    transmitted = [{"doc_id": "42", "sentences": [{"number": 3, "text": "evidence"}]}]
    raw = json.dumps({"label": "SUPPORT", "doc_id": "42", "sentences": [3]})
    assert parse_response(raw, "rag", transmitted)[0]


def test_rationale_requires_every_sentence():
    claim = {"evidence": {"42": [{"label": "SUPPORT", "sentences": [2, 3]}]}}
    full = [{"doc_id": "42", "sentences": [{"number": 2}, {"number": 3}]}]
    partial = [{"doc_id": "42", "sentences": [{"number": 2}]}]
    assert ev.rationale_is_transmitted(claim, full)
    assert not ev.rationale_is_transmitted(claim, partial)

