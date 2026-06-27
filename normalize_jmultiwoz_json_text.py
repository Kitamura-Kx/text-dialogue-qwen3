from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rewrite_jmultiwoz_tsv_to_json_batch import normalize_utterance, validate_dialogues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize generated JSON utterances for TTS-friendly reading.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs/jmultiwoz_rewritten_json"))
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--limit", type=int, default=0, help="Normalize only the first N files. 0 means no limit.")
    return parser.parse_args()


def normalize_dialogues(data: Any) -> tuple[Any, bool]:
    validate_dialogues(data)
    changed = False
    for dialogue in data:
        for turn in dialogue:
            before = turn[1]
            after = normalize_utterance(before)
            if after != before:
                turn[1] = after
                changed = True
    validate_dialogues(data)
    return data, changed


def main() -> None:
    args = parse_args()

    paths: list[Path] = []
    for split in args.splits:
        paths.extend((args.output_root / split).glob("*.json"))
    paths = sorted(
        paths,
        key=lambda path: (
            path.parent.name,
            0 if path.stem.isdigit() else 1,
            int(path.stem) if path.stem.isdigit() else path.stem,
        ),
    )
    if args.limit:
        paths = paths[: args.limit]

    checked = 0
    changed_count = 0
    failed: list[tuple[Path, str]] = []

    for path in paths:
        if path.name.endswith(".partial.json"):
            continue
        checked += 1
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            normalized, changed = normalize_dialogues(data)
        except Exception as exc:  # noqa: BLE001 - report all malformed files.
            failed.append((path, str(exc)))
            print(f"Failed: {path}: {exc}")
            continue

        if changed:
            path.write_text(
                json.dumps(normalized, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            changed_count += 1
            print(f"Normalized: {path}")

    print(f"Done. checked={checked}, changed={changed_count}, failed={len(failed)}")
    if failed:
        print("Failed files:")
        for path, reason in failed[:50]:
            print(f"- {path}: {reason}")
        if len(failed) > 50:
            print(f"... and {len(failed) - 50} more")


if __name__ == "__main__":
    main()
