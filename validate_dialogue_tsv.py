from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


EXPECTED_HEADER = ["dialogue_id", "turn_id", "speaker", "utterance", "goal_domains"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generated dialogue TSV files.")
    parser.add_argument("path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f, delimiter="\t"))

    if not rows:
        raise SystemExit("empty TSV")

    errors: list[str] = []
    header = rows[0]
    if header != EXPECTED_HEADER:
        errors.append(f"header mismatch: expected {EXPECTED_HEADER}, got {header}")

    last_dialogue_id: str | None = None
    expected_turn_id = 0
    for line_no, row in enumerate(rows[1:], start=2):
        if len(row) != len(EXPECTED_HEADER):
            errors.append(f"line {line_no}: expected {len(EXPECTED_HEADER)} columns, got {len(row)}")
            continue

        dialogue_id, turn_id_text, speaker, utterance, _goal_domains = row
        if "\n" in utterance or "\t" in utterance:
            errors.append(f"line {line_no}: utterance contains a newline or tab")
        if speaker not in {"USER", "SYSTEM"}:
            errors.append(f"line {line_no}: speaker must be USER or SYSTEM, got {speaker!r}")

        if dialogue_id != last_dialogue_id:
            last_dialogue_id = dialogue_id
            expected_turn_id = 0

        try:
            turn_id = int(turn_id_text)
        except ValueError:
            errors.append(f"line {line_no}: turn_id is not an integer: {turn_id_text!r}")
            continue

        if turn_id != expected_turn_id:
            errors.append(f"line {line_no}: expected turn_id {expected_turn_id}, got {turn_id}")
        expected_turn_id += 1

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        raise SystemExit(1)

    print(f"OK: {args.path} ({len(rows) - 1} turns)")


if __name__ == "__main__":
    main()
