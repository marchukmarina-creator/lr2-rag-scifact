"""Evaluation-only access to SciFact labels and rationales.

Retrieval and prompt construction live in lr2_core.py and never receive this
module's gold structures, except for the explicitly diagnostic oracle mode.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

LABELS = ("SUPPORT", "CONTRADICT", "NOINFO")


def evidence_rows(claim: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for doc_id, items in (claim.get("evidence") or {}).items():
        for item in items:
            rows.append((str(doc_id), item))
    return rows


def claim_label(claim: dict[str, Any]) -> str | None:
    rows = evidence_rows(claim)
    if not rows:
        return "NOINFO"
    labels = {str(item["label"]).upper() for _, item in rows}
    return next(iter(labels)) if len(labels) == 1 else None


def gold_rationales(claim: dict[str, Any]) -> dict[str, list[list[int]]]:
    result: dict[str, list[list[int]]] = {}
    for doc_id, item in evidence_rows(claim):
        sentence_ids = [int(x) for x in item.get("sentences", [])]
        if sentence_ids:
            result.setdefault(doc_id, []).append(sentence_ids)
    return result


def gold_doc_ids(claim: dict[str, Any]) -> set[str]:
    return set(gold_rationales(claim))


def rationale_is_transmitted(
    claim: dict[str, Any], transmitted: list[dict[str, Any]]
) -> bool:
    sent_by_doc = {
        str(doc["doc_id"]): {int(s["number"]) for s in doc.get("sentences", [])}
        for doc in transmitted
    }
    for doc_id, alternatives in gold_rationales(claim).items():
        available = sent_by_doc.get(doc_id, set())
        if any(set(alt).issubset(available) for alt in alternatives):
            return True
    return False


def link_is_correct(record: dict[str, Any]) -> bool:
    parsed = record.get("parsed") or {}
    if not record.get("parse_ok") or parsed.get("label") != record.get("gold_label"):
        return False
    doc_id = parsed.get("doc_id")
    predicted = {int(x) for x in parsed.get("sentences", [])}
    alternatives = (record.get("gold_rationales") or {}).get(str(doc_id), [])
    return any(set(map(int, alt)).issubset(predicted) for alt in alternatives)


def accuracy(records: Iterable[dict[str, Any]]) -> float:
    rows = list(records)
    if not rows:
        return 0.0
    good = sum(
        bool(r.get("parse_ok"))
        and (r.get("parsed") or {}).get("label") == r.get("gold_label")
        for r in rows
    )
    return good / len(rows)


def class_recall(records: Iterable[dict[str, Any]]) -> dict[str, float]:
    rows = list(records)
    result: dict[str, float] = {}
    for label in LABELS:
        subset = [r for r in rows if r.get("gold_label") == label]
        result[label] = accuracy(subset) if subset else 0.0
    return result


def search_metrics(records: Iterable[dict[str, Any]], k: int) -> dict[str, float]:
    rows = [r for r in records if r.get("gold_label") in ("SUPPORT", "CONTRADICT")]
    hit_values: list[float] = []
    recall_values: list[float] = []
    rr_values: list[float] = []
    rationale_values: list[float] = []
    transmitted_counts: list[int] = []
    prompt_tokens: list[int] = []
    for r in rows:
        relevant = set((r.get("gold_rationales") or {}).keys())
        retrieved = [str(x["doc_id"]) for x in r.get("retrieved", [])[:k]]
        positions = [i + 1 for i, doc_id in enumerate(retrieved) if doc_id in relevant]
        hit_values.append(float(bool(positions)))
        recall_values.append(len(relevant.intersection(retrieved)) / len(relevant))
        rr_values.append(1.0 / min(positions) if positions else 0.0)
        rationale_values.append(float(bool(r.get("rationale_transmitted"))))
        transmitted_counts.append(len(r.get("transmitted", [])))
        prompt_tokens.append(int(r.get("prompt_tokens", 0)))
    n = len(rows) or 1
    return {
        "n": len(rows),
        "hit": sum(hit_values) / n,
        "recall": sum(recall_values) / n,
        "mrr": sum(rr_values) / n,
        "rationale_transmitted": sum(rationale_values) / n,
        "mean_transmitted_docs": sum(transmitted_counts) / n,
        "mean_prompt_tokens": sum(prompt_tokens) / n,
    }


def paired_changes(
    before: Iterable[dict[str, Any]], after: Iterable[dict[str, Any]]
) -> dict[str, int]:
    a = {int(r["claim_id"]): r for r in before}
    b = {int(r["claim_id"]): r for r in after}

    def correct(r: dict[str, Any]) -> bool:
        return bool(r.get("parse_ok")) and (r.get("parsed") or {}).get("label") == r.get("gold_label")

    fixed = spoiled = both_correct = both_wrong = 0
    for claim_id in sorted(set(a).intersection(b)):
        ca, cb = correct(a[claim_id]), correct(b[claim_id])
        fixed += int(not ca and cb)
        spoiled += int(ca and not cb)
        both_correct += int(ca and cb)
        both_wrong += int(not ca and not cb)
    return {
        "n": len(set(a).intersection(b)),
        "fixed": fixed,
        "spoiled": spoiled,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
    }


def status_counts(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(r.get("status", "unknown")) for r in records))


def known_metric_example() -> dict[str, float]:
    relevant = {"a", "b"}
    top3 = ["x", "y", "a"]
    return {
        "hit3": float(bool(relevant.intersection(top3))),
        "recall3": len(relevant.intersection(top3)) / len(relevant),
        "rr3": 1 / 3,
        "hit1": float(bool(relevant.intersection(top3[:1]))),
        "recall1": len(relevant.intersection(top3[:1])) / len(relevant),
        "rr1": 0.0,
    }
