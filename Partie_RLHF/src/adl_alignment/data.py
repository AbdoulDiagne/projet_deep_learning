from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

from datasets import load_dataset


def write_jsonl(rows: Iterable[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def prepare_hh_preferences(dataset_name: str, split: str, max_samples: int, seed: int) -> list[dict]:
    """Convert HH-RLHF examples into chosen/rejected preference rows."""
    dataset = load_dataset(dataset_name, split=split)
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    rows = []

    for idx in indices[:max_samples]:
        example = dataset[idx]
        rows.append(
            {
                "prompt": _extract_prompt(example["chosen"]),
                "chosen": example["chosen"],
                "rejected": example["rejected"],
            }
        )
    return rows


def _extract_prompt(text: str) -> str:
    marker = "\n\nAssistant:"
    if marker in text:
        return text.split(marker, maxsplit=1)[0].strip()
    return text[:500].strip()


def load_ethics_subset(dataset_name: str, subset: str, split: str, max_samples: int | None = None):
    """Load an ETHICS subset for evaluation only."""
    dataset = load_dataset(dataset_name, subset, split=split)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset
