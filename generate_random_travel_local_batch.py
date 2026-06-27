from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from normalize_dialogue_json import normalize_file
from validate_dialogue_json import validate_path


DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DEFAULT_SYSTEM = "あなたは日本語の対話データを丁寧に生成・編集するアシスタントです。"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate many local dialogue JSON files while loading the model only once.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--file-count", type=int, default=1000)
    parser.add_argument("--dialogues-per-file", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--system", default=DEFAULT_SYSTEM)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("file_count", "dialogues_per_file", "batch_size", "max_retries", "max_tokens"):
        if getattr(args, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.base_seed < 0:
        raise SystemExit("--base-seed must be non-negative")


def output_path(output_dir: Path, index: int) -> Path:
    return output_dir / f"random_travel_dialogues.{index:04d}.json"


def existing_output_is_valid(path: Path, expected_count: int) -> bool:
    if not path.is_file():
        return False
    try:
        validate_path(path, expected_count)
    except (json.JSONDecodeError, ValueError):
        return False
    return True


def main() -> None:
    args = parse_args()
    validate_args(args)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Run: bash scripts/setup_local.sh") from exc

    prompt = args.prompt_file.read_text(encoding="utf-8")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pending: list[int] = []
    for index in range(1, args.file_count + 1):
        path = output_path(args.output_dir, index)
        if args.resume and existing_output_is_valid(path, args.dialogues_per_file):
            print(f"Skip valid: {path}")
        else:
            pending.append(index)

    if not pending:
        print("All requested output files are already valid.")
        return

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

    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": prompt},
    ]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    attempts = {index: 0 for index in pending}
    failures: list[int] = []
    completed = args.file_count - len(pending)

    while pending:
        batch_indices = pending[: args.batch_size]
        del pending[: args.batch_size]
        seed = args.base_seed + batch_indices[0] + attempts[batch_indices[0]] * args.file_count
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        inputs = tokenizer(
            [formatted_prompt] * len(batch_indices),
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

        for index, output in zip(batch_indices, outputs, strict=True):
            final_path = output_path(args.output_dir, index)
            partial_path = final_path.with_suffix(".partial.json")
            partial_path.write_text(output.rstrip() + "\n", encoding="utf-8")
            try:
                normalize_file(partial_path)
                validate_path(partial_path, args.dialogues_per_file)
            except (json.JSONDecodeError, ValueError) as exc:
                attempts[index] += 1
                print(
                    f"Retry {index:04d} ({attempts[index]}/{args.max_retries}): {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                if attempts[index] < args.max_retries:
                    pending.append(index)
                else:
                    failures.append(index)
                continue

            partial_path.replace(final_path)
            completed += 1
            print(f"Completed {completed}/{args.file_count}: {final_path}", flush=True)

    if failures:
        joined = ", ".join(f"{index:04d}" for index in failures)
        raise SystemExit(f"Failed after retries: {joined}")

    print(
        f"Completed: {args.file_count} files, "
        f"{args.file_count * args.dialogues_per_file} dialogues in {args.output_dir}",
    )


if __name__ == "__main__":
    main()
