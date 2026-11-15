"""LR2: reproducible SciFact RAG experiments and report generation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
import urllib.request
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

import lr2_eval as ev
from lr2_core import (
    DenseIndex,
    PromptBuilder,
    append_jsonl,
    call_model,
    document_text,
    download,
    load_json,
    load_jsonl,
    parse_response,
    save_json,
    sha256_file,
)

ROOT = Path(__file__).resolve().parent


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_config(path: str | Path) -> dict[str, Any]:
    return load_json(Path(path) if Path(path).is_absolute() else ROOT / path)


def package_versions() -> dict[str, str]:
    names = ["numpy", "sentence-transformers", "transformers", "huggingface-hub", "openai", "mlflow", "matplotlib"]
    result = {"python": platform.python_version()}
    for name in names:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "missing"
    return result


def git_info(require_clean: bool = False) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    try:
        commit = run("rev-parse", "HEAD")
        status = run("status", "--porcelain")
    except Exception as exc:
        if require_clean:
            raise RuntimeError("Run the final experiment from the cloned Git repository") from exc
        return {"commit": None, "clean": None, "status": "git unavailable"}
    if require_clean and status:
        raise RuntimeError("Final run requires a clean committed repository. Commit source changes first.\n" + status)
    return {"commit": commit, "clean": not bool(status), "status": status}


def extract_scifact(archive: Path, raw_dir: Path) -> dict[str, Path]:
    wanted = {"corpus.jsonl", "claims_train.jsonl", "claims_dev.jsonl"}
    raw_dir.mkdir(parents=True, exist_ok=True)
    found: dict[str, Path] = {}
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            name = Path(member.name).name
            if name not in wanted or name.startswith("._") or not member.isfile():
                continue
            source = bundle.extractfile(member)
            if source is None:
                continue
            destination = raw_dir / name
            with destination.open("wb") as out:
                shutil.copyfileobj(source, out)
            found[name] = destination
    missing = wanted - set(found)
    if missing:
        raise RuntimeError(f"SciFact archive is missing: {sorted(missing)}")
    return found


def split_claims(claims: list[dict[str, Any]], per_class: int) -> tuple[list[int], list[int], list[int]]:
    by_class: dict[str, list[int]] = {label: [] for label in ev.LABELS}
    excluded: list[int] = []
    for claim in claims:
        label = ev.claim_label(claim)
        if label is None:
            excluded.append(int(claim["id"]))
            continue
        by_class[label].append(int(claim["id"]))
    dev: list[int] = []
    holdout: list[int] = []
    for label in ev.LABELS:
        ordered = sorted(
            by_class[label],
            key=lambda i: hashlib.sha256(f"scifact:{i}".encode()).hexdigest(),
        )
        dev.extend(ordered[:per_class])
        holdout.extend(ordered[per_class:])
    return dev, holdout, excluded


def prepare(config_path: str) -> None:
    config = read_config(config_path)
    raw_dir = ROOT / "data" / "raw"
    generated = ROOT / "data" / "generated"
    archive = raw_dir / "data.tar.gz"
    if not archive.exists():
        print("Downloading the official SciFact release...")
        download(config["data_url"], archive)
    actual_archive_hash = sha256_file(archive)
    if actual_archive_hash != config["data_archive_sha256"]:
        raise RuntimeError(f"SciFact archive SHA-256 mismatch: {actual_archive_hash}")
    files = extract_scifact(archive, raw_dir)
    claims_dev = load_jsonl(files["claims_dev.jsonl"])
    dev_ids, holdout_ids, excluded = split_claims(claims_dev, int(config["dev_per_class"]))
    claims_by_id = {int(x["id"]): x for x in claims_dev}
    dev_counts = Counter(ev.claim_label(claims_by_id[i]) for i in dev_ids)
    holdout_counts = Counter(ev.claim_label(claims_by_id[i]) for i in holdout_ids)
    if dict(dev_counts) != {"SUPPORT": 30, "CONTRADICT": 30, "NOINFO": 30}:
        raise AssertionError(f"Unexpected dev split: {dev_counts}")
    if dict(holdout_counts) != {"SUPPORT": 94, "CONTRADICT": 34, "NOINFO": 82}:
        raise AssertionError(f"Unexpected holdout split: {holdout_counts}")

    generated.mkdir(parents=True, exist_ok=True)
    save_json(generated / "split.json", {"dev": dev_ids, "holdout": holdout_ids, "excluded": excluded})

    corpus_rows = load_jsonl(files["corpus.jsonl"])
    corpus_rows.sort(key=lambda x: int(x["doc_id"]))
    texts = [document_text(x) for x in corpus_rows]
    print(f"Encoding {len(texts)} SciFact abstracts on CPU...")
    from sentence_transformers import SentenceTransformer
    from huggingface_hub import HfApi

    encoder = SentenceTransformer(config["encoder"], device="cpu")
    encoder.max_seq_length = int(config["encoder_max_length"])
    tokenized = encoder.tokenizer(texts, add_special_tokens=True, truncation=False, padding=False)
    lengths = [len(x) for x in tokenized["input_ids"]]
    truncated_count = sum(x > int(config["encoder_max_length"]) for x in lengths)
    matrix = encoder.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    index_dir = ROOT / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / "embeddings.npy", matrix)
    save_json(index_dir / "doc_ids.json", [str(x["doc_id"]) for x in corpus_rows])
    try:
        revision = HfApi().model_info(config["encoder"]).sha
    except Exception:
        revision = "unavailable"
    matrix_hash = sha256_file(index_dir / "embeddings.npy")
    passport = {
        "encoder": config["encoder"],
        "revision": revision,
        "dimension": int(matrix.shape[1]),
        "max_length": int(config["encoder_max_length"]),
        "normalized": True,
        "document_prefix": "",
        "query_prefix": config.get("query_instruction", ""),
        "corpus_size": int(matrix.shape[0]),
        "truncated_count": truncated_count,
        "truncated_fraction": truncated_count / len(texts),
        "matrix_sha256": matrix_hash,
        "doc_ids_sha256": sha256_file(index_dir / "doc_ids.json"),
    }
    save_json(index_dir / "passport.json", passport)
    manifest = {
        "source": config["data_url"],
        "archive_sha256": actual_archive_hash,
        "files": {name: {"sha256": sha256_file(path), "rows": len(load_jsonl(path))} for name, path in files.items()},
        "label_rule": "empty evidence -> NOINFO; otherwise the unique evidence label",
        "split_rule": "within each class sort by SHA-256('scifact:<id>'); first 30 to dev",
        "dev_ids": dev_ids,
        "holdout_ids": holdout_ids,
        "excluded_mixed_label_ids": excluded,
        "dev_counts": dict(dev_counts),
        "holdout_counts": dict(holdout_counts),
        "holdout_used_in_lr2": False,
        "index_passport": passport,
    }
    save_json(generated / "manifest.json", manifest)
    print(json.dumps({"dev": dict(dev_counts), "holdout": dict(holdout_counts), "index": passport}, ensure_ascii=False, indent=2))


def load_assets(config: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[int, dict[str, Any]], list[int]]:
    corpus_rows = load_jsonl(ROOT / "data" / "raw" / "corpus.jsonl")
    corpus = {str(x["doc_id"]): x for x in corpus_rows}
    claims = {int(x["id"]): x for x in load_jsonl(ROOT / "data" / "raw" / "claims_dev.jsonl")}
    split = load_json(ROOT / "data" / "generated" / "split.json")
    return corpus, claims, [int(x) for x in split["dev"]]


def check(config_path: str) -> None:
    config = read_config(config_path)
    corpus, claims, dev_ids = load_assets(config)
    example = ev.known_metric_example()
    assert example == {"hit3": 1.0, "recall3": 0.5, "rr3": 1 / 3, "hit1": 0.0, "recall1": 0.0, "rr1": 0.0}
    sample_claim = next(claims[i] for i in dev_ids if ev.claim_label(claims[i]) != "NOINFO")
    rationales = ev.gold_rationales(sample_claim)
    doc_id, alternatives = next(iter(rationales.items()))
    chosen = alternatives[0]
    transmitted = [{"doc_id": doc_id, "sentences": [{"number": x, "text": "x"} for x in chosen]}]
    assert ev.rationale_is_transmitted(sample_claim, transmitted)
    if chosen:
        broken = [{"doc_id": doc_id, "sentences": [{"number": x, "text": "x"} for x in chosen[:-1]]}]
        if len(chosen) > 1:
            assert not ev.rationale_is_transmitted(sample_claim, broken)
    good = json.dumps({"label": "SUPPORT", "doc_id": doc_id, "sentences": chosen})
    ok, _, _ = parse_response(good, "rag", transmitted)
    assert ok
    assert not parse_response('{"label":"UNKNOWN","doc_id":null,"sentences":[]}', "rag", transmitted)[0]
    assert not parse_response('{"label":"NOINFO","doc_id":null,"sentences":[],"extra":1}', "rag", transmitted)[0]
    assert not parse_response('{"label":"SUPPORT","doc_id":"missing","sentences":[0]}', "rag", transmitted)[0]

    index = DenseIndex(ROOT, config)
    vector = index.encode_query("A short scientific claim.")
    assert abs(float(vector @ vector) - 1.0) < 1e-4
    retrieved = index.search(str(sample_claim["claim"]), 5)
    assert len(retrieved) == 5 and all(retrieved[i]["score"] >= retrieved[i + 1]["score"] for i in range(4))
    builder = PromptBuilder(ROOT, config, corpus)
    built = builder.rag(str(sample_claim["claim"]), retrieved, 3, int(config["context_budget"]))
    assert built["prompt_tokens"] <= int(config["context_budget"])
    print("All pre-experiment checks passed.")


def server_ready(config: dict[str, Any]) -> None:
    url = config["base_url"].rstrip("/") + "/models"
    with urllib.request.urlopen(url, timeout=10) as response:
        payload = json.load(response)
    ids = [str(x.get("id")) for x in payload.get("data", [])]
    if config["generator_model"] not in ids:
        raise RuntimeError(f"Expected model {config['generator_model']}; server exposes {ids}")


def oracle_documents(claim: dict[str, Any], corpus: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for doc_id, alternatives in ev.gold_rationales(claim).items():
        chosen = alternatives[0]
        source = corpus[doc_id]
        result.append({
            "doc_id": doc_id,
            "title": source.get("title", ""),
            "sentences": [{"number": i, "text": source["abstract"][i]} for i in chosen],
        })
        break
    return result


def build_request(
    group: str,
    mode: str,
    k: int | None,
    claim: dict[str, Any],
    index: DenseIndex,
    builder: PromptBuilder,
    corpus: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    retrieved = index.search(str(claim["claim"]), int(config["retrieval_top_n"])) if mode == "rag" else []
    if mode == "llm_only":
        prompt = builder.llm_only(str(claim["claim"]))
    elif mode == "rag":
        prompt = builder.rag(str(claim["claim"]), retrieved, int(k), int(config["context_budget"]))
    elif mode == "oracle":
        prompt = builder.oracle(str(claim["claim"]), oracle_documents(claim, corpus), int(config["context_budget"]))
    else:
        raise ValueError(mode)
    label = ev.claim_label(claim)
    job_id = f"{group}:{claim['id']}"
    return {
        "job_id": job_id,
        "group": group,
        "mode": mode,
        "k": k,
        "claim_id": int(claim["id"]),
        "claim": str(claim["claim"]),
        "gold_label": label,
        "gold_rationales": ev.gold_rationales(claim),
        "retrieved": retrieved,
        "transmitted": prompt["transmitted"],
        "rationale_transmitted": ev.rationale_is_transmitted(claim, prompt["transmitted"]) if label != "NOINFO" else None,
        "prompt": prompt["prompt"],
        "messages": prompt["messages"],
        "prompt_tokens": prompt["prompt_tokens"],
    }


def execute_request(request: dict[str, Any], config: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    result = {k: v for k, v in request.items() if k != "messages"}
    result["started_at"] = utc_now()
    try:
        response = call_model(config, schema, request["messages"])
        result.update(response)
        ok, parsed, error = parse_response(response["raw_response"], request["mode"], request["transmitted"])
        result.update({"parse_ok": ok, "parsed": parsed, "parse_error": error, "status": "ok" if ok else "parse_error"})
    except Exception as exc:
        result.update({
            "raw_response": None,
            "parsed": None,
            "parse_ok": False,
            "parse_error": None,
            "status": "request_error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })
    result["finished_at"] = utc_now()
    return result


def group_plan(dev_ids: list[int], claims: dict[int, dict[str, Any]]) -> list[tuple[str, str, int | None, list[int]]]:
    positive = [i for i in dev_ids if ev.claim_label(claims[i]) in ("SUPPORT", "CONTRADICT")]
    return [
        ("exp1_llm_only", "llm_only", None, dev_ids),
        ("exp1_rag_k3", "rag", 3, dev_ids),
        ("exp1_oracle", "oracle", None, positive),
        ("exp2_rag_k1", "rag", 1, dev_ids),
        ("exp2_rag_k5", "rag", 5, dev_ids),
    ]


def run_experiments(config_path: str) -> None:
    config = read_config(config_path)
    server_ready(config)
    git = git_info(require_clean=True)
    corpus, claims, dev_ids = load_assets(config)
    index = DenseIndex(ROOT, config)
    builder = PromptBuilder(ROOT, config, corpus)
    schema = load_json(ROOT / "configs" / "response_schema.json")
    run_dir = ROOT / "runs" / config["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"
    requests_path = run_dir / "requests.jsonl"
    existing = {r["job_id"] for r in load_jsonl(results_path)} if results_path.exists() else set()
    metadata = {
        "run_id": config["run_id"],
        "started_at": utc_now(),
        "finished_at": None,
        "git": git,
        "versions": package_versions(),
        "generator_model": config["generator_model"],
        "generator_revision": config["generator_revision"],
        "index_passport": load_json(ROOT / "index" / "passport.json"),
        "config_sha256": sha256_file(ROOT / config_path),
        "manifest_sha256": sha256_file(ROOT / "data" / "generated" / "manifest.json"),
        "planned_calls": 420,
        "holdout_used": False,
    }
    save_json(run_dir / "config.json", config)
    save_json(run_dir / "metadata.json", metadata)
    subprocess.run([sys.executable, "-m", "pip", "freeze"], cwd=ROOT, check=True, stdout=(run_dir / "requirements.lock.txt").open("w", encoding="utf-8"))

    import mlflow

    mlflow.set_tracking_uri((ROOT / "mlflow.db").resolve().as_uri().replace("file://", "sqlite:///"))
    mlflow.set_experiment("lr2")
    with mlflow.start_run(run_name=config["run_id"]) as parent:
        metadata["mlflow_parent_run_id"] = parent.info.run_id
        save_json(run_dir / "metadata.json", metadata)
        mlflow.set_tags({"lab": "LR2", "dataset": "SciFact", "git_commit": git["commit"]})
        mlflow.log_params({
            "encoder": config["encoder"], "generator": config["generator_model"],
            "context_budget": config["context_budget"], "temperature": config["temperature"],
            "concurrency": config["concurrency"], "dev_size": len(dev_ids),
        })

        # Warm-up is explicit and excluded from the 420 measured calls.
        train_claims = load_jsonl(ROOT / "data" / "raw" / "claims_train.jsonl")[:2]
        warmup_path = run_dir / "warmup.jsonl"
        if not warmup_path.exists():
            for claim in train_claims:
                prompt = builder.llm_only(str(claim["claim"]))
                warm = {"claim_id": int(claim["id"]), "started_at": utc_now()}
                try:
                    warm.update(call_model(config, schema, prompt["messages"]))
                    warm["status"] = "ok"
                except Exception as exc:
                    warm.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
                append_jsonl(warmup_path, warm)

        for group, mode, k, ids in group_plan(dev_ids, claims):
            requests = [build_request(group, mode, k, claims[i], index, builder, corpus, config) for i in ids]
            missing = [x for x in requests if x["job_id"] not in existing]
            for request in missing:
                append_jsonl(requests_path, {key: value for key, value in request.items() if key != "messages"})
            print(f"{group}: {len(missing)} missing of {len(requests)}")
            group_results: list[dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=int(config["concurrency"])) as pool:
                futures = {pool.submit(execute_request, req, config, schema): req["job_id"] for req in missing}
                for number, future in enumerate(as_completed(futures), 1):
                    record = future.result()
                    append_jsonl(results_path, record)
                    existing.add(record["job_id"])
                    group_results.append(record)
                    if number % 10 == 0 or number == len(missing):
                        print(f"  completed {number}/{len(missing)}")
            all_group = [r for r in load_jsonl(results_path) if r["group"] == group]
            with mlflow.start_run(run_name=group, nested=True):
                mlflow.log_params({"mode": mode, "k": -1 if k is None else k, "claims": len(ids)})
                mlflow.log_metrics({"accuracy": ev.accuracy(all_group), "parse_failures": sum(not r.get("parse_ok") for r in all_group)})

        final_rows = load_jsonl(results_path)
        metadata["finished_at"] = utc_now()
        metadata["saved_calls"] = len(final_rows)
        metadata["status_counts"] = ev.status_counts(final_rows)
        save_json(run_dir / "metadata.json", metadata)
        mlflow.log_metrics({"saved_calls": len(final_rows), "failed_calls": sum(r.get("status") != "ok" for r in final_rows)})
        mlflow.log_artifact(str(results_path))
        mlflow.log_artifact(str(run_dir / "metadata.json"))
    print(f"Measured calls saved: {len(load_jsonl(results_path))}")


def rows_for_group(rows: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return sorted([r for r in rows if r["group"] == name], key=lambda x: int(x["claim_id"]))


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_error_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rag = rows_for_group(rows, "exp1_rag_k3")
    candidates: list[tuple[str, Any]] = [
        ("evidence_not_retrieved", lambda r: r["gold_label"] != "NOINFO" and not set(r["gold_rationales"]).intersection(x["doc_id"] for x in r["retrieved"][:3])),
        ("retrieved_not_transmitted", lambda r: r["gold_label"] != "NOINFO" and set(r["gold_rationales"]).intersection(x["doc_id"] for x in r["retrieved"][:3]) and not r.get("rationale_transmitted")),
        ("transmitted_but_wrong", lambda r: r.get("rationale_transmitted") and (not r.get("parse_ok") or (r.get("parsed") or {}).get("label") != r["gold_label"])),
        ("noinfo_error", lambda r: r["gold_label"] == "NOINFO" and (not r.get("parse_ok") or (r.get("parsed") or {}).get("label") != "NOINFO")),
    ]
    selected: list[dict[str, Any]] = []
    used: set[int] = set()
    for category, predicate in candidates:
        match = next((r for r in rag if int(r["claim_id"]) not in used and predicate(r)), None)
        if match is None:
            selected.append({"category": category, "found": False})
        else:
            used.add(int(match["claim_id"]))
            selected.append({
                "category": category,
                "found": True,
                "claim_id": match["claim_id"],
                "claim": match["claim"],
                "gold_label": match["gold_label"],
                "retrieved": match["retrieved"],
                "transmitted": match["transmitted"],
                "raw_response": match.get("raw_response"),
                "parsed": match.get("parsed"),
                "rationale_transmitted": match.get("rationale_transmitted"),
            })
    return selected


def analyze(config_path: str) -> None:
    config = read_config(config_path)
    run_dir = ROOT / "runs" / config["run_id"]
    rows = load_jsonl(run_dir / "results.jsonl")
    if len(rows) != 420:
        raise RuntimeError(f"Expected 420 records, found {len(rows)}")
    groups = {name: rows_for_group(rows, name) for name, *_ in group_plan([], {})} if False else {
        name: rows_for_group(rows, name) for name in ["exp1_llm_only", "exp1_rag_k3", "exp1_oracle", "exp2_rag_k1", "exp2_rag_k5"]
    }
    report_dir = ROOT / "reports" / "generated"
    figures = report_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    exp1_table = []
    for name in ("exp1_llm_only", "exp1_rag_k3"):
        recall = ev.class_recall(groups[name])
        exp1_table.append({
            "condition": name,
            "n": len(groups[name]),
            "accuracy": ev.accuracy(groups[name]),
            "recall_SUPPORT": recall["SUPPORT"],
            "recall_CONTRADICT": recall["CONTRADICT"],
            "recall_NOINFO": recall["NOINFO"],
            "correct_link": (sum(ev.link_is_correct(r) for r in groups[name] if r["gold_label"] != "NOINFO") / 60) if name == "exp1_rag_k3" else "",
        })
    positive_ids = {r["claim_id"] for r in groups["exp1_oracle"]}
    oracle_table = []
    for name in ("exp1_llm_only", "exp1_rag_k3", "exp1_oracle"):
        subset = [r for r in groups[name] if r["claim_id"] in positive_ids]
        oracle_table.append({"condition": name, "n": len(subset), "accuracy": ev.accuracy(subset)})
    k_table = []
    for k, name in ((1, "exp2_rag_k1"), (3, "exp1_rag_k3"), (5, "exp2_rag_k5")):
        search = ev.search_metrics(groups[name], k)
        k_table.append({
            "k": k,
            **search,
            "accuracy": ev.accuracy(groups[name]),
            "correct_link": sum(ev.link_is_correct(r) for r in groups[name] if r["gold_label"] != "NOINFO") / 60,
        })
    paired = {
        "llm_to_rag3": ev.paired_changes(groups["exp1_llm_only"], groups["exp1_rag_k3"]),
        "rag1_to_rag3": ev.paired_changes(groups["exp2_rag_k1"], groups["exp1_rag_k3"]),
        "rag3_to_rag5": ev.paired_changes(groups["exp1_rag_k3"], groups["exp2_rag_k5"]),
    }
    errors = select_error_cases(rows)
    summary = {
        "run_id": config["run_id"],
        "records": len(rows),
        "status_counts": ev.status_counts(rows),
        "experiment1": exp1_table,
        "oracle_same_60": oracle_table,
        "paired": paired,
        "k_comparison": k_table,
        "error_cases": errors,
        "holdout_used": False,
    }
    save_json(report_dir / "summary.json", summary)
    save_json(report_dir / "error_cases.json", errors)
    csv_write(report_dir / "experiment1.csv", exp1_table)
    csv_write(report_dir / "oracle.csv", oracle_table)
    csv_write(report_dir / "k_comparison.csv", k_table)

    import matplotlib.pyplot as plt

    ks = [x["k"] for x in k_table]
    plt.figure(figsize=(7, 4))
    for key, label in (("hit", "Hit@k"), ("recall", "Recall@k"), ("mrr", "MRR@k"), ("rationale_transmitted", "Evidence in prompt")):
        plt.plot(ks, [x[key] for x in k_table], marker="o", label=label)
    plt.xticks(ks); plt.ylim(0, 1.02); plt.xlabel("k"); plt.ylabel("fraction"); plt.grid(alpha=.25); plt.legend(); plt.tight_layout()
    plt.savefig(figures / "retrieval_by_k.png", dpi=180); plt.close()
    plt.figure(figsize=(7, 4))
    labels = ["LLM-only", "RAG k=1", "RAG k=3", "RAG k=5", "Oracle (60)"]
    values = [ev.accuracy(groups["exp1_llm_only"]), ev.accuracy(groups["exp2_rag_k1"]), ev.accuracy(groups["exp1_rag_k3"]), ev.accuracy(groups["exp2_rag_k5"]), ev.accuracy(groups["exp1_oracle"])]
    plt.bar(labels, values); plt.ylim(0, 1); plt.ylabel("Accuracy"); plt.xticks(rotation=20, ha="right"); plt.tight_layout()
    plt.savefig(figures / "answer_accuracy.png", dpi=180); plt.close()

    def pct(x: Any) -> str:
        return f"{100 * float(x):.1f}%"

    report = [
        "# ЛР2. Базовый RAG для SciFact",
        "",
        "## Прогноз и условия",
        "",
        (ROOT / "configs" / "prediction.md").read_text(encoding="utf-8"),
        "",
        "Holdout не использовался. Все 420 запланированных вызовов входят в знаменатели, включая ошибки вызова и разбора.",
        "",
        "## Эксперимент 1",
        "",
        "| Режим | n | Accuracy | Recall SUPPORT | Recall CONTRADICT | Recall NOINFO | Верная ссылка |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for x in exp1_table:
        link = "—" if x["correct_link"] == "" else pct(x["correct_link"])
        report.append(f"| {x['condition']} | {x['n']} | {pct(x['accuracy'])} | {pct(x['recall_SUPPORT'])} | {pct(x['recall_CONTRADICT'])} | {pct(x['recall_NOINFO'])} | {link} |")
    p = paired["llm_to_rag3"]
    report += ["", f"Попарно LLM-only → RAG k=3: исправлено {p['fixed']}, испорчено {p['spoiled']} из {p['n']} утверждений.", "", "Oracle сравнивается только на одинаковых 60 утверждениях SUPPORT/CONTRADICT:", "", "| Режим | n | Accuracy |", "|---|---:|---:|"]
    for x in oracle_table:
        report.append(f"| {x['condition']} | {x['n']} | {pct(x['accuracy'])} |")
    report += ["", "## Эксперимент 2", "", "| k | Hit@k | Recall@k | MRR@k | Обоснование передано | Ср. документов | Ср. токенов | Accuracy | Верная ссылка |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for x in k_table:
        report.append(f"| {x['k']} | {pct(x['hit'])} | {pct(x['recall'])} | {pct(x['mrr'])} | {pct(x['rationale_transmitted'])} | {x['mean_transmitted_docs']:.2f} | {x['mean_prompt_tokens']:.1f} | {pct(x['accuracy'])} | {pct(x['correct_link'])} |")
    report += ["", "## Разбор ошибок", ""]
    for case in errors:
        if not case["found"]:
            report.append(f"- **{case['category']}**: такого случая в RAG k=3 нет; это явно зафиксировано вместо подмены другим типом ошибки.")
        else:
            report.append(f"- **{case['category']}**, claim {case['claim_id']} ({case['gold_label']}): {case['claim']} Сырой ответ: `{case['raw_response']}`.")
    report += ["", "## Вывод", "", "Итоговый вывод формулируется по таблицам: отдельно проверяется качество поиска, попадание обоснования в промпт и способность генератора правильно использовать уже переданный текст. Рост Recall сам по себе не доказывает улучшение конечного ответа.", "", "## Кодогенерация", "", "Каркас и проверки подготовлены с помощью OpenAI Codex. Все численные результаты пересчитаны скриптом из `runs/lr2_final/results.jsonl`; до опытов выполнены автоматические проверки метрик, контекста и разбора JSON."]
    (report_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    import mlflow
    mlflow.set_tracking_uri((ROOT / "mlflow.db").resolve().as_uri().replace("file://", "sqlite:///"))
    metadata = load_json(run_dir / "metadata.json")
    with mlflow.start_run(run_id=metadata["mlflow_parent_run_id"]):
        mlflow.log_metrics({
            "llm_only_accuracy": ev.accuracy(groups["exp1_llm_only"]),
            "rag_k3_accuracy": ev.accuracy(groups["exp1_rag_k3"]),
            "oracle_accuracy_on_60": ev.accuracy(groups["exp1_oracle"]),
            "rag_k3_correct_link": k_table[1]["correct_link"],
            "rag_k3_recall": k_table[1]["recall"],
        })
        mlflow.log_artifacts(str(report_dir), artifact_path="reports")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def pack(config_path: str) -> None:
    config = read_config(config_path)
    output = ROOT / f"lr2_{config['run_id']}_artifacts.zip"
    include = [ROOT / "runs" / config["run_id"], ROOT / "reports" / "generated", ROOT / "data" / "generated", ROOT / "index", ROOT / "mlflow.db"]
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for source in include:
            if source.is_file():
                bundle.write(source, source.relative_to(ROOT))
            elif source.exists():
                for path in source.rglob("*"):
                    if path.is_file():
                        bundle.write(path, path.relative_to(ROOT))
    print(f"ARTIFACTS: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "check", "run", "analyze", "pack"])
    parser.add_argument("--config", default="configs/base.json")
    args = parser.parse_args()
    if args.command == "prepare": prepare(args.config)
    elif args.command == "check": check(args.config)
    elif args.command == "run": run_experiments(args.config)
    elif args.command == "analyze": analyze(args.config)
    elif args.command == "pack": pack(args.config)


if __name__ == "__main__":
    main()
