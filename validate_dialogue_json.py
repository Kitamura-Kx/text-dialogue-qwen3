from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generated dialogue JSON files.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--expected-count", type=int, default=5)
    return parser.parse_args()


def load_dialogues(data: Any) -> list[Any]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("dialogues"), list):
        return data["dialogues"]
    raise ValueError("top-level JSON must be an array or an object with a 'dialogues' array")


def validate_turn(turn: Any, dialogue_index: int, turn_index: int) -> str | None:
    if not isinstance(turn, list) or len(turn) != 2:
        return f"dialogue {dialogue_index}, turn {turn_index}: turn must be [speaker, utterance]"
    speaker, utterance = turn
    if speaker not in {"SYSTEM", "USER"}:
        return f"dialogue {dialogue_index}, turn {turn_index}: speaker must be SYSTEM or USER"
    if not isinstance(utterance, str) or not utterance.strip():
        return f"dialogue {dialogue_index}, turn {turn_index}: utterance must be a non-empty string"
    return None


def validate_path(path: Path, expected_count: int = 5) -> tuple[int, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    dialogues = load_dialogues(data)

    errors: list[str] = []
    if len(dialogues) != expected_count:
        errors.append(f"expected {expected_count} dialogues, got {len(dialogues)}")

    for dialogue_index, dialogue in enumerate(dialogues):
        if not isinstance(dialogue, list):
            errors.append(f"dialogue {dialogue_index}: dialogue must be an array of turns")
            continue
        if len(dialogue) < 6:
            errors.append(f"dialogue {dialogue_index}: expected at least 6 turns, got {len(dialogue)}")
        for turn_index, turn in enumerate(dialogue):
            error = validate_turn(turn, dialogue_index, turn_index)
            if error:
                errors.append(error)

    if errors:
        raise ValueError("\n".join(errors))

    turn_count = sum(len(dialogue) for dialogue in dialogues if isinstance(dialogue, list))
    return len(dialogues), turn_count


def main() -> None:
    args = parse_args()
    try:
        dialogue_count, turn_count = validate_path(args.path, args.expected_count)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"OK: {args.path} ({dialogue_count} dialogues, {turn_count} turns)")


if __name__ == "__main__":
    main()
