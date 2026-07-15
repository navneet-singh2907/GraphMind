"""Small dependency-free BM25 index over canonical GraphMind chunks."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path


DEFAULT_CHUNKS_PATH = Path("data/processed/document_chunks.jsonl")
TOKEN_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_+.-]*")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


class KeywordIndex:
    def __init__(self, records: list[dict], k1: float = 1.5, b: float = 0.75) -> None:
        self.records = records
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize(record.get("content", "")) for record in records]
        self.term_frequencies = [Counter(tokens) for tokens in self.tokens]
        self.document_lengths = [len(tokens) for tokens in self.tokens]
        self.average_length = sum(self.document_lengths) / len(self.document_lengths) if records else 0
        self.document_frequency: Counter[str] = Counter()
        for tokens in self.tokens:
            self.document_frequency.update(set(tokens))

    @classmethod
    def from_jsonl(cls, path: Path = DEFAULT_CHUNKS_PATH) -> "KeywordIndex":
        if not path.exists():
            return cls([])
        with path.open("r", encoding="utf-8") as handle:
            return cls([json.loads(line) for line in handle if line.strip()])

    def search(
        self,
        query: str,
        k: int = 8,
        collection: str | None = None,
        source_type: str | None = None,
    ) -> list[dict]:
        if not self.records:
            return []
        query_tokens = tokenize(query)
        scores: list[tuple[float, int]] = []
        total = len(self.records)

        for index, (record, frequencies, length) in enumerate(
            zip(self.records, self.term_frequencies, self.document_lengths)
        ):
            if collection and record.get("collection") != collection:
                continue
            if source_type and record.get("source_type") != source_type:
                continue
            score = 0.0
            for token in query_tokens:
                frequency = frequencies.get(token, 0)
                if not frequency:
                    continue
                document_frequency = self.document_frequency[token]
                inverse_frequency = math.log(1 + (total - document_frequency + 0.5) / (document_frequency + 0.5))
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / max(self.average_length, 1)
                )
                score += inverse_frequency * frequency * (self.k1 + 1) / denominator
            if score > 0:
                scores.append((score, index))

        return [
            {**self.records[index], "score": round(score, 6), "retriever": "keyword"}
            for score, index in sorted(scores, reverse=True)[:k]
        ]
