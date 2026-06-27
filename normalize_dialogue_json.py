from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SPEAKERS = {"SYSTEM", "USER"}
TRAILING_INTERRUPTION_DASHES = re.compile(r"[、。！？?!]?[―—–]+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize common formatting errors in generated dialogue JSON.",
    )
    parser.add_argument("path", type=Path)
    return parser.parse_args()


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines.pop()
    return "\n".join(lines).strip()


def decode_json_values(text: str) -> list[Any]:
    decoder = json.JSONDecoder()
    values: list[Any] = []
    position = 0

    while position < len(text):
        while position < len(text) and (text[position].isspace() or text[position] == ","):
            position += 1
        if position >= len(text):
            break
        value, position = decoder.raw_decode(text, position)
        values.append(value)

    if not values:
        raise ValueError("generated output does not contain JSON")
    return values


def is_turn_pair(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and value[0] in SPEAKERS
        and isinstance(value[1], str)
        and bool(value[1].strip())
    )


def split_turns(value: Any, dialogue_index: int, turn_index: int) -> list[list[str]]:
    if is_turn_pair(value):
        speaker, utterance = value
        return [[speaker, normalize_utterance(utterance)]]

    if isinstance(value, list) and len(value) >= 4 and len(value) % 2 == 0:
        turns = [value[index : index + 2] for index in range(0, len(value), 2)]
        if all(is_turn_pair(turn) for turn in turns):
            return [
                [speaker, normalize_utterance(utterance)]
                for speaker, utterance in turns
            ]

    raise ValueError(
        f"dialogue {dialogue_index}, turn {turn_index}: cannot normalize turn into [speaker, utterance]",
    )


def normalize_utterance(utterance: str) -> str:
    return TRAILING_INTERRUPTION_DASHES.sub("、", utterance.strip())


def normalize_dialogue(value: Any, dialogue_index: int) -> list[list[str]]:
    if not isinstance(value, list):
        raise ValueError(f"dialogue {dialogue_index}: dialogue must be an array")

    normalized: list[list[str]] = []
    for turn_index, turn in enumerate(value):
        normalized.extend(split_turns(turn, dialogue_index, turn_index))
    return normalized


def extract_dialogues(values: list[Any]) -> list[Any]:
    if len(values) > 1:
        return values

    value = values[0]
    if isinstance(value, dict) and isinstance(value.get("dialogues"), list):
        return value["dialogues"]
    if not isinstance(value, list):
        raise ValueError("top-level JSON must be an array")

    if value and all(is_turn_pair(turn) for turn in value):
        return [value]
    return value


def normalize_file(path: Path) -> None:
    text = strip_code_fence(path.read_text(encoding="utf-8"))
    values = decode_json_values(text)
    dialogues = extract_dialogues(values)
    normalized = [
        normalize_dialogue(dialogue, dialogue_index)
        for dialogue_index, dialogue in enumerate(dialogues)
    ]
    path.write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Normalized {path}")


def main() -> None:
    args = parse_args()
    normalize_file(args.path)


if __name__ == "__main__":
    main()
