from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DEFAULT_SYSTEM = "あなたは日本語の対話データを丁寧に編集するアシスタントです。"
REQUIRED_FIRST_TURN = ["SYSTEM", "こんにちは、何か旅行の予定はありますか？"]
SPEAKERS = {"SYSTEM", "USER"}
TRAILING_INTERRUPTION_DASHES = re.compile(r"[、。！？?!]?[―—–]+$")
SLASH_DATE = re.compile(r"(?<![0-9０-９])([0-9０-９]{1,2})[\/／]([0-9０-９]{1,2})(?![0-9０-９])")
COLON_TIME = re.compile(r"(?<![0-9０-９])([0-9０-９]{1,2})[:：]([0-9０-９]{2})(?![0-9０-９])")
MISSING_MINUTE_SUFFIX_TIME = re.compile(r"(?<![0-9０-９])([0-9０-９]{1,2})時([0-9０-９]{1,2})(?!分|[0-9０-９])")
COUNT_FOR_TTS = re.compile(r"(?<![0-9０-９])([0-9０-９]{1,2})(名|人|泊)")
FULLWIDTH_DIGITS_BEFORE_DATE_TIME_UNIT = re.compile(r"[０-９]+(?=[月日時分])")
FULLWIDTH_DIGIT_TRANS = str.maketrans("０１２３４５６７８９", "0123456789")


@dataclass(frozen=True)
class RewriteTask:
    split: str
    source_path: Path
    output_path: Path
    ordinal: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite JMultiWOZ per-dialogue TSV files into one-dialogue JSON files with a local Qwen model.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, default=Path("jmultiwoz-tsv-by-dialogue"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/jmultiwoz_rewritten_json"))
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=2000)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--system", default=DEFAULT_SYSTEM)
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N pending files. 0 means no limit.")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("batch_size", "max_retries", "max_tokens"):
        if getattr(args, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.base_seed < 0:
        raise SystemExit("--base-seed must be non-negative")
    if args.limit < 0:
        raise SystemExit("--limit must be zero or positive")


def natural_sort_key(path: Path) -> tuple[int, int | str]:
    if path.stem.isdigit():
        return (0, int(path.stem))
    return (1, path.stem)


def collect_tasks(input_root: Path, output_root: Path, splits: list[str]) -> list[RewriteTask]:
    tasks: list[RewriteTask] = []
    ordinal = 0
    for split in splits:
        split_dir = input_root / split
        if not split_dir.is_dir():
            raise SystemExit(f"Input split directory does not exist: {split_dir}")
        for source_path in sorted(split_dir.glob("*.tsv"), key=natural_sort_key):
            output_path = output_root / split / f"{source_path.stem}.json"
            tasks.append(
                RewriteTask(
                    split=split,
                    source_path=source_path,
                    output_path=output_path,
                    ordinal=ordinal,
                ),
            )
            ordinal += 1
    return tasks


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


def ascii_digits(text: str) -> str:
    return text.translate(FULLWIDTH_DIGIT_TRANS)


def number_to_kanji(number: int) -> str:
    digits = "零一二三四五六七八九"
    if 0 <= number < 10:
        return digits[number]
    if number == 10:
        return "十"
    if 10 < number < 20:
        return "十" + digits[number % 10]
    if number < 100:
        tens, ones = divmod(number, 10)
        return digits[tens] + "十" + (digits[ones] if ones else "")
    return str(number)


def normalize_for_tts(text: str) -> str:
    text = SLASH_DATE.sub(
        lambda match: f"{ascii_digits(match.group(1))}月{ascii_digits(match.group(2))}日",
        text,
    )
    text = COLON_TIME.sub(
        lambda match: f"{ascii_digits(match.group(1))}時{ascii_digits(match.group(2))}分",
        text,
    )
    text = MISSING_MINUTE_SUFFIX_TIME.sub(
        lambda match: f"{ascii_digits(match.group(1))}時{ascii_digits(match.group(2))}分",
        text,
    )
    text = COUNT_FOR_TTS.sub(
        lambda match: f"{number_to_kanji(int(ascii_digits(match.group(1))))}{match.group(2)}",
        text,
    )
    text = FULLWIDTH_DIGITS_BEFORE_DATE_TIME_UNIT.sub(lambda match: ascii_digits(match.group(0)), text)
    return text


def normalize_utterance(utterance: str) -> str:
    text = TRAILING_INTERRUPTION_DASHES.sub("、", utterance.strip())
    return normalize_for_tts(text)


def normalize_turn(turn: Any, dialogue_index: int, turn_index: int) -> list[str]:
    if not is_turn_pair(turn):
        raise ValueError(
            f"dialogue {dialogue_index}, turn {turn_index}: turn must be [speaker, utterance]",
        )
    speaker, utterance = turn
    return [speaker, normalize_utterance(utterance)]


def extract_single_dialogue(value: Any) -> list[Any]:
    if isinstance(value, dict) and isinstance(value.get("dialogues"), list):
        value = value["dialogues"]

    if not isinstance(value, list):
        raise ValueError("top-level JSON must be an array")

    if value and all(is_turn_pair(turn) for turn in value):
        return value

    if value and all(
        isinstance(chunk, list) and chunk and all(is_turn_pair(turn) for turn in chunk)
        for chunk in value
    ):
        return [turn for chunk in value for turn in chunk]

    if len(value) != 1:
        raise ValueError(f"top-level array must contain exactly one dialogue, got {len(value)}")

    dialogue = value[0]
    if not isinstance(dialogue, list):
        raise ValueError("dialogue must be an array of turns")
    return dialogue


def normalize_generated_json(text: str) -> list[list[list[str]]]:
    values = decode_json_values(strip_code_fence(text))
    if len(values) != 1:
        raise ValueError(f"expected one JSON value, got {len(values)}")

    dialogue = extract_single_dialogue(values[0])
    normalized_dialogue = [
        normalize_turn(turn, dialogue_index=0, turn_index=turn_index)
        for turn_index, turn in enumerate(dialogue)
    ]
    normalized = [normalized_dialogue]
    validate_dialogues(normalized)
    return normalized


def validate_dialogues(dialogues: Any) -> None:
    if not isinstance(dialogues, list):
        raise ValueError("top-level JSON must be an array")
    if len(dialogues) != 1:
        raise ValueError(f"expected exactly 1 dialogue, got {len(dialogues)}")

    dialogue = dialogues[0]
    if not isinstance(dialogue, list):
        raise ValueError("dialogue must be an array of turns")
    if len(dialogue) < 4:
        raise ValueError(f"expected at least 4 turns, got {len(dialogue)}")
    if dialogue[0] != REQUIRED_FIRST_TURN:
        raise ValueError("first turn must be the fixed SYSTEM greeting")

    for turn_index, turn in enumerate(dialogue):
        if not is_turn_pair(turn):
            raise ValueError(f"turn {turn_index}: turn must be [speaker, utterance]")


def existing_output_is_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        validate_dialogues(data)
    except (json.JSONDecodeError, ValueError):
        return False
    return True


def build_prompt(prompt_template: str, source_path: Path) -> str:
    dialogue_tsv = source_path.read_text(encoding="utf-8").strip()
    if "{{DIALOGUE_TSV}}" in prompt_template:
        return prompt_template.replace("{{DIALOGUE_TSV}}", dialogue_tsv)
    return f"{prompt_template.rstrip()}\n\n入力TSV:\n{dialogue_tsv}"


def write_normalized_output(path: Path, text: str) -> None:
    normalized = normalize_generated_json(text)
    path.write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    validate_args(args)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Local generation environment is not installed. Run the setup script from text-dialogue-qwen3.") from exc

    prompt_template = args.prompt_file.read_text(encoding="utf-8")
    tasks = collect_tasks(args.input_root, args.output_root, args.splits)

    pending: list[RewriteTask] = []
    for task in tasks:
        if args.resume and existing_output_is_valid(task.output_path):
            print(f"Skip valid: {task.output_path}")
        else:
            pending.append(task)

    if args.limit:
        pending = pending[: args.limit]

    if not pending:
        print("All requested output files are already valid.")
        return

    for task in pending:
        task.output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading local model once: {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        local_files_only=True,
    )
    model.eval()

    attempts = {task.source_path: 0 for task in pending}
    failures: list[RewriteTask] = []
    completed = len(tasks) - len(pending)
    total_requested = completed + len(pending)

    while pending:
        batch_tasks = pending[: args.batch_size]
        del pending[: args.batch_size]

        seed = args.base_seed + batch_tasks[0].ordinal + attempts[batch_tasks[0].source_path] * len(tasks)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        formatted_prompts: list[str] = []
        for task in batch_tasks:
            user_prompt = build_prompt(prompt_template, task.source_path)
            messages = [
                {"role": "system", "content": args.system},
                {"role": "user", "content": user_prompt},
            ]
            formatted_prompts.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                ),
            )

        inputs = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
        ).to(model.device)
        input_length = inputs.input_ids.shape[1]

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else None,
                top_p=args.top_p,
            )

        outputs = tokenizer.batch_decode(
            generated[:, input_length:],
            skip_special_tokens=True,
        )

        for task, output in zip(batch_tasks, outputs, strict=True):
            final_path = task.output_path
            partial_path = final_path.with_suffix(".partial.json")
            partial_path.write_text(output.rstrip() + "\n", encoding="utf-8")

            try:
                write_normalized_output(partial_path, output)
            except (json.JSONDecodeError, ValueError) as exc:
                attempts[task.source_path] += 1
                print(
                    f"Retry {task.split}/{task.source_path.name} "
                    f"({attempts[task.source_path]}/{args.max_retries}): {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                if attempts[task.source_path] < args.max_retries:
                    pending.append(task)
                else:
                    failures.append(task)
                continue

            partial_path.replace(final_path)
            completed += 1
            print(f"Completed {completed}/{total_requested}: {final_path}", flush=True)

    if failures:
        joined = ", ".join(f"{task.split}/{task.source_path.name}" for task in failures)
        raise SystemExit(f"Failed after retries: {joined}")

    print(f"Completed: {total_requested} files in {args.output_root}")


if __name__ == "__main__":
    main()
