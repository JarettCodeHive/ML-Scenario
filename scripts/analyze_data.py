"""Analyze a CSV in chunks and write a Markdown report."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from cluster_c.data_analysis import analyze_csv, render_markdown


LOGGER = logging.getLogger("analyze_data")
DEFAULT_INPUT = Path("data/input/term_deals.csv")
DEFAULT_OUTPUT = Path("DATA_ANALYSIS_REPORT.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input CSV path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output Markdown path")
    parser.add_argument("--chunk-size", type=int, default=100_000, help="Rows processed per chunk")
    parser.add_argument("--unique-limit", type=int, default=10_000, help="Exact distinct-value tracking cap")
    parser.add_argument("--top-values", type=int, default=10, help="Frequent values shown per bounded column")
    args = parser.parse_args()

    LOGGER.info("Analyzing %s in chunks of %s rows", args.input, f"{args.chunk_size:,}")
    result = analyze_csv(
        args.input,
        chunk_size=args.chunk_size,
        unique_limit=args.unique_limit,
        top_value_limit=args.top_values,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(result, args.top_values), encoding="utf-8")
    LOGGER.info(
        "Analysis completed: %s rows, %d columns, %s null/blank cells",
        f"{result.row_count:,}",
        len(result.columns),
        f"{result.total_missing:,}",
    )
    LOGGER.info("Markdown report written to %s", output)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    try:
        main()
    except Exception:
        LOGGER.exception("Data analysis failed")
        raise SystemExit(1)
