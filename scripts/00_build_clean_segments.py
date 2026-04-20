from __future__ import annotations

import argparse
import json
from pathlib import Path

import jsonlines

MODE_SET = {"teaching", "bridge", "persona", "commentary"}


def infer_mode_from_path(path: Path) -> str:
    parts = set(path.parts)
    for mode in MODE_SET:
        if mode in parts:
            return mode
    raise ValueError(f"Cannot infer mode from path: {path}")


def build_source_file(root: Path, file_path: Path) -> str:
    rel = file_path.relative_to(root)
    return str(rel).replace("\\", "/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        default="/root/persona-skill-factory/data",
        help="Root directory containing mode folders",
    )
    parser.add_argument(
        "--output",
        default="/root/persona-skill-factory/artifacts/clean/all_clean_segments.jsonl",
        help="Output JSONL path",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(input_root.glob("*/clean/**/*.json"))
    total_files = 0
    total_segments = 0

    with jsonlines.open(output_path, "w") as writer:
        for fp in files:
            mode = infer_mode_from_path(fp)
            source_file = build_source_file(input_root, fp)

            with fp.open("r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError(f"Expected list in {fp}, got {type(data)}")

            total_files += 1

            for item in data:
                paragraph_id = item.get("paragraph_id")
                text = (item.get("text") or "").strip()
                if not text:
                    continue

                segment = {
                    "source_file": source_file,
                    "corpus_mode": mode,
                    "segment_id": f"p{int(paragraph_id):04d}" if paragraph_id is not None else "p0000",
                    "text": text,
                    "speaker": "target",
                    "start_sec": item.get("start_time"),
                    "end_sec": item.get("end_time"),
                    "metadata": {
                        "source_segment_indices": item.get("source_segment_indices", []),
                        "paragraph_id": paragraph_id,
                        "filename": fp.name,
                        "series": fp.parent.name,
                    },
                }
                writer.write(segment)
                total_segments += 1

    print(f"done: files={total_files}, segments={total_segments}, output={output_path}")


if __name__ == "__main__":
    main()
