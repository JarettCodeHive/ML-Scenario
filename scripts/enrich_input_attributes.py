"""Stream new required attributes into an existing term-deals CSV."""

from __future__ import annotations

import argparse
import csv
import logging
import os
import random
from pathlib import Path

from cluster_c.config import (
    ADDITIONAL_SOURCE_COLUMNS,
    CX_UPSELL_GROUP_VALUES,
    XCAT_FLAG_VALUES,
)


LOGGER = logging.getLogger("enrich_input_attributes")


def _generated_values(rng: random.Random) -> dict[str, str]:
    attribution = rng.uniform(-100.0, 100.0)
    if abs(attribution) < 0.000001:
        attribution = 0.000001
    return {
        "CX_UPSELL_GROUP": rng.choice(CX_UPSELL_GROUP_VALUES),
        "XCAT_FLG": rng.choice(XCAT_FLAG_VALUES),
        "DV_ATTRIBUTION_CD": f"{attribution:.6f}",
    }


def enrich(path: Path, seed: int = 42) -> int:
    """Atomically add or fill required attributes without loading the CSV."""
    temporary = path.with_name(f".{path.name}.enrich.tmp")
    rng = random.Random(seed)
    row_count = 0
    try:
        with path.open("r", newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if not reader.fieldnames:
                raise ValueError(f"CSV has no header: {path}")
            fieldnames = [
                *reader.fieldnames,
                *(column for column in ADDITIONAL_SOURCE_COLUMNS if column not in reader.fieldnames),
            ]
            with temporary.open("w", newline="", encoding="utf-8", buffering=1024 * 1024) as output:
                writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
                writer.writeheader()
                for row_count, row in enumerate(reader, start=1):
                    generated = _generated_values(rng)
                    for column in ADDITIONAL_SOURCE_COLUMNS:
                        if not str(row.get(column, "")).strip():
                            row[column] = generated[column]
                    writer.writerow(row)
        os.replace(temporary, path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return row_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for path in args.paths:
        rows = enrich(path, args.seed)
        LOGGER.info("Updated %s with %s rows", path, f"{rows:,}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    try:
        main()
    except Exception:
        LOGGER.exception("Input enrichment failed")
        raise SystemExit(1)
