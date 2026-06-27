from __future__ import annotations

import argparse
from pathlib import Path

from rewrite_jmultiwoz_tsv_to_json_batch import natural_sort_key, write_normalized_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair generated *.partial.json files and promote valid files to *.json.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs/jmultiwoz_rewritten_json"))
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--limit", type=int, default=0, help="Repair only the first N partial files. 0 means no limit.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing final .json files.")
    parser.add_argument("--delete-repaired", action="store_true", help="Delete partial files after successful repair.")
    return parser.parse_args()


def sort_key(path: Path) -> tuple[str, int, int | str]:
    return (path.parent.name, *natural_sort_key(path.with_suffix("").with_suffix("")))


def main() -> None:
    args = parse_args()

    partials: list[Path] = []
    for split in args.splits:
        split_dir = args.output_root / split
        partials.extend(split_dir.glob("*.partial.json"))

    partials = sorted(partials, key=sort_key)
    if args.limit:
        partials = partials[: args.limit]

    repaired = 0
    skipped = 0
    failed: list[tuple[Path, str]] = []

    for partial_path in partials:
        final_path = partial_path.with_name(partial_path.name.replace(".partial.json", ".json"))
        if final_path.exists() and not args.overwrite:
            skipped += 1
            print(f"Skip existing final: {final_path}")
            continue

        try:
            write_normalized_output(final_path, partial_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - report all malformed partials.
            failed.append((partial_path, str(exc)))
            print(f"Failed: {partial_path}: {exc}")
            continue

        repaired += 1
        print(f"Repaired: {partial_path} -> {final_path}")
        if args.delete_repaired:
            partial_path.unlink()

    print(f"Done. repaired={repaired}, skipped={skipped}, failed={len(failed)}")
    if failed:
        print("Failed files:")
        for path, reason in failed[:50]:
            print(f"- {path}: {reason}")
        if len(failed) > 50:
            print(f"... and {len(failed) - 50} more")


if __name__ == "__main__":
    main()
