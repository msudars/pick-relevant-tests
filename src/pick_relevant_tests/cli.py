from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import (
    DEFAULT_CSV_DIR,
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SHEET_NAME,
)
from .system import BatteryDataSystem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local battery metadata mapping and retrieval system.")
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sheet-name", default=DEFAULT_SHEET_NAME)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--query", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--skip-semantic", action="store_true")
    parser.add_argument("--use-ollama", action="store_true")
    parser.add_argument("--ollama-model", default="llama3")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    system = BatteryDataSystem(
        metadata_path=args.metadata_path,
        csv_dir=args.csv_dir,
        output_dir=args.output_dir,
        sheet_name=args.sheet_name,
        model_name=args.model_name,
    )
    system.run(build_semantic=not args.skip_semantic)

    if args.query:
        if args.use_ollama:
            parsed = system.parse_query_with_llm(args.query, model=args.ollama_model)
            semantic_text = parsed.pop("semantic", None)
            filtered = system.query_tests(**parsed)
            if semantic_text:
                semantic_hits = system.semantic_search(semantic_text, k=args.top_k)
                result = filtered.merge(semantic_hits, on="test_id", how="inner")
            else:
                result = filtered
        else:
            result = system.hybrid_query(args.query, k=args.top_k)

        columns = [
            column
            for column in [
                "test_id",
                "cell_id",
                "lot_number",
                "chemistry",
                "regime",
                "temperature",
                "c_rate",
                "status",
                "score",
                "explanation",
                "csv_path",
            ]
            if column in result.columns
        ]
        print("\nHYBRID QUERY RESULTS")
        if result.empty:
            print("No matching tests found.")
        else:
            print(result[columns].to_string(index=False))


def run() -> int:
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
    return 0
