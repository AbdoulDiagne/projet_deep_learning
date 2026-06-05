from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from adl_alignment.config import load_config
from adl_alignment.data import prepare_hh_preferences, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rlhf.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rows = prepare_hh_preferences(
        dataset_name=cfg["data"]["preference_dataset"],
        split=cfg["data"]["preference_split"],
        max_samples=cfg["data"]["max_preference_samples"],
        seed=cfg["seed"],
    )
    write_jsonl(rows, cfg["data"]["processed_preferences"])
    print(f"Wrote {len(rows)} preference rows to {cfg['data']['processed_preferences']}")


if __name__ == "__main__":
    main()
