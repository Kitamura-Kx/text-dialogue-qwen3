from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_PROMPT_FILE = Path(__file__).resolve().parent / "prompts" / "rewrite_kyowa.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Japanese dialogue text with Qwen3-30B-A3B-Instruct-2507.",
    )
    parser.add_argument("--backend", choices=["openai", "local"], default="openai")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--prompt", help="Prompt text. If omitted, --prompt-file is used.")
    parser.add_argument("--input-file", type=Path, help="TSV or text file inserted into {{DIALOGUE_TSV}}.")
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--system", default="あなたは日本語の対話データを丁寧に生成・編集するアシスタントです。")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--json-mode",
        action="store_true",
        help="Ask an OpenAI-compatible backend to return a JSON object when supported.",
    )
    return parser.parse_args()


def build_prompt(args: argparse.Namespace) -> str:
    if args.prompt:
        template = args.prompt
    else:
        template = args.prompt_file.read_text(encoding="utf-8")

    if args.input_file:
        input_text = args.input_file.read_text(encoding="utf-8").strip()
        if "{{DIALOGUE_TSV}}" in template:
            return template.replace("{{DIALOGUE_TSV}}", input_text)
        return f"{template.rstrip()}\n\n入力:\n{input_text}"

    return template


def generate_with_openai(args: argparse.Namespace, prompt: str) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("openai is not installed. Run: uv sync") from exc

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    request: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": args.system},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    if args.seed is not None:
        request["seed"] = args.seed
    if args.json_mode:
        request["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**request)
    return response.choices[0].message.content or ""


def generate_with_local(args: argparse.Namespace, prompt: str) -> str:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("local backend dependencies are not installed. Run: uv sync --extra local") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
    )
    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=args.max_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature if args.temperature > 0 else None,
            top_p=args.top_p,
        )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def write_output(output: str, output_file: Path | None) -> None:
    if output_file is None:
        print(output)
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(output.rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {output_file}", file=sys.stderr)


def main() -> None:
    args = parse_args()
    prompt = build_prompt(args)
    if args.backend == "openai":
        output = generate_with_openai(args, prompt)
    else:
        output = generate_with_local(args, prompt)
    write_output(output, args.output_file)


if __name__ == "__main__":
    main()
