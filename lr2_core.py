"""Gold-blind retrieval, context construction, parsing and model calls for LR2."""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

LABELS = {"SUPPORT", "CONTRADICT", "NOINFO"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "lr2-scifact-course"})
    with urllib.request.urlopen(request, timeout=180) as response, temp.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    temp.replace(destination)


def document_text(doc: dict[str, Any]) -> str:
    abstract = " ".join(str(x) for x in doc.get("abstract", []))
    return f"{doc.get('title', '')}\n{abstract}".strip()


class DenseIndex:
    def __init__(self, root: Path, config: dict[str, Any]):
        from sentence_transformers import SentenceTransformer

        self.root = root
        self.config = config
        self.encoder = SentenceTransformer(config["encoder"], device="cpu")
        self.encoder.max_seq_length = int(config["encoder_max_length"])
        self.matrix = np.load(root / "index" / "embeddings.npy", mmap_mode="r")
        self.doc_ids = load_json(root / "index" / "doc_ids.json")

    def encode_query(self, claim: str) -> np.ndarray:
        text = self.config.get("query_instruction", "") + claim
        vector = self.encoder.encode(
            [text], batch_size=1, convert_to_numpy=True, normalize_embeddings=True
        )[0]
        return np.asarray(vector, dtype=np.float32)

    def search(self, claim: str, top_n: int = 5) -> list[dict[str, Any]]:
        started = time.perf_counter()
        query = self.encode_query(claim)
        scores = np.asarray(self.matrix @ query)
        candidate = np.argpartition(-scores, min(top_n, len(scores) - 1))[:top_n]
        order = candidate[np.argsort(-scores[candidate])]
        elapsed = time.perf_counter() - started
        return [
            {
                "doc_id": str(self.doc_ids[int(i)]),
                "score": float(scores[int(i)]),
                "rank": rank,
                "search_seconds": elapsed,
            }
            for rank, i in enumerate(order, 1)
        ]


class PromptBuilder:
    def __init__(self, root: Path, config: dict[str, Any], corpus: dict[str, dict[str, Any]]):
        from transformers import AutoTokenizer

        self.root = root
        self.config = config
        self.corpus = corpus
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["generator_model"], revision=config.get("generator_revision") or None
        )
        self.prompts = {
            name: (root / "prompts" / f"{name}.txt").read_text(encoding="utf-8").strip()
            for name in ("llm_only", "rag", "oracle")
        }

    def token_count(self, messages: list[dict[str, str]]) -> int:
        ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=False
        )
        return len(ids)

    def render(self, messages: list[dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @staticmethod
    def _doc_block(doc: dict[str, Any], sentence_limit: int | None = None) -> tuple[str, list[dict[str, Any]]]:
        sentences = [str(x) for x in doc.get("abstract", [])]
        if sentence_limit is not None:
            sentences = sentences[:sentence_limit]
        entries = [{"number": i, "text": text} for i, text in enumerate(sentences)]
        lines = [f"DOC {doc['doc_id']}\nTITLE: {doc.get('title', '')}"]
        lines.extend(f"[{x['number']}] {x['text']}" for x in entries)
        return "\n".join(lines), entries

    def _messages(self, system: str, claim: str, documents_text: str | None = None) -> list[dict[str, str]]:
        user = f"CLAIM: {claim}"
        if documents_text is not None:
            user += "\n\nDOCUMENTS:\n" + documents_text
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def llm_only(self, claim: str) -> dict[str, Any]:
        messages = self._messages(self.prompts["llm_only"], claim)
        return {
            "messages": messages,
            "prompt": self.render(messages),
            "prompt_tokens": self.token_count(messages),
            "transmitted": [],
        }

    def rag(self, claim: str, retrieved: list[dict[str, Any]], k: int, budget: int) -> dict[str, Any]:
        blocks: list[str] = []
        transmitted: list[dict[str, Any]] = []
        for item in retrieved[:k]:
            doc = self.corpus[str(item["doc_id"])]
            full_block, full_entries = self._doc_block(doc)
            candidate = "\n\n".join(blocks + [full_block])
            if self.token_count(self._messages(self.prompts["rag"], claim, candidate)) <= budget:
                blocks.append(full_block)
                transmitted.append(
                    {"doc_id": str(doc["doc_id"]), "title": doc.get("title", ""), "sentences": full_entries}
                )
                continue

            accepted_block = None
            accepted_entries: list[dict[str, Any]] = []
            for count in range(1, len(doc.get("abstract", [])) + 1):
                partial_block, partial_entries = self._doc_block(doc, count)
                candidate = "\n\n".join(blocks + [partial_block])
                if self.token_count(self._messages(self.prompts["rag"], claim, candidate)) <= budget:
                    accepted_block, accepted_entries = partial_block, partial_entries
                else:
                    break
            if accepted_block is not None:
                blocks.append(accepted_block)
                transmitted.append(
                    {"doc_id": str(doc["doc_id"]), "title": doc.get("title", ""), "sentences": accepted_entries}
                )
            break

        docs_text = "\n\n".join(blocks)
        messages = self._messages(self.prompts["rag"], claim, docs_text)
        tokens = self.token_count(messages)
        if tokens > budget:
            raise AssertionError(f"context budget exceeded: {tokens} > {budget}")
        return {
            "messages": messages,
            "prompt": self.render(messages),
            "prompt_tokens": tokens,
            "transmitted": transmitted,
        }

    def oracle(self, claim: str, evidence_docs: list[dict[str, Any]], budget: int) -> dict[str, Any]:
        blocks: list[str] = []
        transmitted: list[dict[str, Any]] = []
        for doc in evidence_docs:
            entries = [{"number": int(x["number"]), "text": str(x["text"])} for x in doc["sentences"]]
            lines = [f"DOC {doc['doc_id']}\nTITLE: {doc.get('title', '')}"]
            lines.extend(f"[{x['number']}] {x['text']}" for x in entries)
            blocks.append("\n".join(lines))
            transmitted.append({"doc_id": str(doc["doc_id"]), "title": doc.get("title", ""), "sentences": entries})
        messages = self._messages(self.prompts["oracle"], claim, "\n\n".join(blocks))
        tokens = self.token_count(messages)
        if tokens > budget:
            raise AssertionError(f"oracle context budget exceeded: {tokens} > {budget}")
        return {
            "messages": messages,
            "prompt": self.render(messages),
            "prompt_tokens": tokens,
            "transmitted": transmitted,
        }


def parse_response(raw: str, mode: str, transmitted: list[dict[str, Any]]) -> tuple[bool, dict[str, Any] | None, str | None]:
    try:
        value = json.loads(raw)
    except Exception as exc:
        return False, None, f"invalid JSON: {exc}"
    if not isinstance(value, dict) or set(value) != {"label", "doc_id", "sentences"}:
        return False, None, "response must contain exactly label, doc_id, sentences"
    if value["label"] not in LABELS:
        return False, None, "unknown label"
    if value["doc_id"] is not None and not isinstance(value["doc_id"], str):
        return False, None, "doc_id must be a string or null"
    if not isinstance(value["sentences"], list) or any(
        isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in value["sentences"]
    ):
        return False, None, "sentences must be non-negative integers"
    if value["label"] == "NOINFO" and (value["doc_id"] is not None or value["sentences"]):
        return False, None, "NOINFO must use null doc_id and empty sentences"
    if mode in {"rag", "oracle"} and value["label"] != "NOINFO":
        allowed = {
            str(doc["doc_id"]): {int(s["number"]) for s in doc.get("sentences", [])}
            for doc in transmitted
        }
        if value["doc_id"] not in allowed:
            return False, None, "cited document was not transmitted"
        if not value["sentences"] or not set(value["sentences"]).issubset(allowed[value["doc_id"]]):
            return False, None, "cited sentence was not transmitted"
    return True, value, None


def call_model(config: dict[str, Any], schema: dict[str, Any], messages: list[dict[str, str]]) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(
        base_url=config["base_url"],
        api_key=config.get("api_key", "none"),
        timeout=float(config.get("request_timeout_seconds", 180)),
        max_retries=0,
    )
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=config["generator_model"],
        messages=messages,
        max_tokens=int(config["max_output_tokens"]),
        temperature=float(config["temperature"]),
        top_p=float(config["top_p"]),
        seed=int(config["seed"]),
        response_format=schema,
        extra_body={"top_k": -1, "min_p": 0, "repetition_penalty": 1.0},
    )
    elapsed = time.perf_counter() - started
    usage = response.usage
    return {
        "raw_response": response.choices[0].message.content or "",
        "generation_seconds": elapsed,
        "usage": {
            "prompt_tokens": int(usage.prompt_tokens) if usage else None,
            "completion_tokens": int(usage.completion_tokens) if usage else None,
            "total_tokens": int(usage.total_tokens) if usage else None,
        },
    }
