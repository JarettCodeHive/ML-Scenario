"""Generate a large, reproducible CSV from the 10-row issue-data template."""

from __future__ import annotations

import argparse
import csv
import io
import logging
import os
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

from cluster_c.config import (
    ADDITIONAL_SOURCE_COLUMNS,
    CX_UPSELL_GROUP_VALUES,
    RAW_COLUMNS,
    XCAT_FLAG_VALUES,
)

LOGGER = logging.getLogger("generate_large_input")
DEFAULT_TEMPLATE = Path("data/input/term_deals copy.csv")
DEFAULT_OUTPUT = Path("data/input/term_deals.csv")
DEFAULT_TARGET_BYTES = 1_030_000_000
TERM_CHOICES = ("1 Year", "3 Year", "5 Year")
TERM_WEIGHTS = (0.23, 0.57, 0.20)


def encode_csv_row(values: list[str]) -> bytes:
    buffer = io.StringIO(newline="")
    csv.writer(buffer, lineterminator="\n").writerow(values)
    return buffer.getvalue().encode("utf-8")


def add_years(value: datetime, years: int) -> datetime:
    try:
        return value.replace(year=value.year + years)
    except ValueError:
        return value.replace(month=2, day=28, year=value.year + years)


def generated_row(
    template: dict[str, str],
    row_number: int,
    rng: random.Random,
) -> dict[str, str]:
    row = template.copy()
    term = rng.choices(TERM_CHOICES, weights=TERM_WEIGHTS, k=1)[0]
    years = int(term.split()[0])
    row["TERM_BUCKET"] = term

    # Retain a small, deterministic amount of the null behavior from the issue
    # sample while keeping nearly all identifiers unique for model testing.
    row["SUBSCRIPTION_REFERENCE_ID"] = (
        "" if row_number % 1_000 == 0 else f"SR-LARGE-{row_number:09d}"
    )
    customer_number = (row_number % 1_000_000) + 1
    row["CAV_END_CUSTOMER_BU_ID"] = str(7_000_000 + customer_number)
    base_name = template["CAV_END_CUSTOMER_BU_NAME"] or "UNKNOWN CUSTOMER"
    row["CAV_END_CUSTOMER_BU_NAME"] = f"{base_name} {customer_number:06d}"
    row["UCD_C4_ACCOUNT_ID"] = str(100_000_000 + customer_number)

    if row_number % 2_000 == 0:
        row["TERM_START_DATE"] = ""
        row["TERM_END_DATE"] = ""
        fiscal_year = 2026
        fiscal_month = 1
    else:
        start = datetime(2020, 1, 1) + timedelta(days=(row_number * 37) % (6 * 365))
        end = add_years(start, years) - timedelta(seconds=1)
        row["TERM_START_DATE"] = start.strftime("%Y-%m-%d %H:%M:%S.000000000")
        row["TERM_END_DATE"] = end.strftime("%Y-%m-%d %H:%M:%S.000000000")
        fiscal_year = start.year + (1 if start.month >= 8 else 0)
        fiscal_month = (start.month - 8) % 12 + 1

    row["FISCAL_YEAR"] = str(fiscal_year)
    row["FISCAL_PERIOD_ID"] = str(fiscal_month)
    row["FISCAL_QUARTER_ID"] = str((fiscal_month - 1) // 3 + 1)

    try:
        source_annualized = abs(float(template["ANNUALIZED_NET_PRICE_USD_AMT"]))
    except (TypeError, ValueError):
        source_annualized = 0.0
    annualized = max(10_000.0, source_annualized * 100 + (row_number % 50_000))
    annualized *= 0.85 + rng.random() * 0.30
    if row_number % 5_000 == 0:
        annualized *= -1
    row["ANNUALIZED_NET_PRICE_USD_AMT"] = f"{annualized:.10f}"
    row["BOOKINGS_NET_PRICE_USD_AMT"] = f"{annualized * years:.10f}"
    row["CX_UPSELL_GROUP"] = rng.choice(CX_UPSELL_GROUP_VALUES)
    row["XCAT_FLG"] = rng.choice(XCAT_FLAG_VALUES)
    dv_attribution = rng.uniform(-100.0, 100.0)
    if abs(dv_attribution) < 0.000001:
        dv_attribution = 0.000001
    row["DV_ATTRIBUTION_CD"] = f"{dv_attribution:.6f}"
    return row


def generate(template_path: Path, output_path: Path, target_bytes: int, seed: int) -> None:
    if target_bytes <= 0:
        raise ValueError("target_bytes must be positive")
    if template_path.resolve() == output_path.resolve():
        raise ValueError("Template and output paths must be different")

    with template_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        templates = list(reader)
        fieldnames = reader.fieldnames or []
    if not templates:
        raise ValueError(f"Template has no data rows: {template_path}")
    missing = sorted((set(RAW_COLUMNS) - set(ADDITIONAL_SOURCE_COLUMNS)) - set(fieldnames))
    if missing:
        raise ValueError(f"Template is missing required columns: {missing}")
    fieldnames = [*fieldnames, *(column for column in ADDITIONAL_SOURCE_COLUMNS if column not in fieldnames)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    rng = random.Random(seed)
    header = encode_csv_row(fieldnames)
    bytes_written = len(header)
    row_count = 0
    next_progress = 100_000_000
    started = time.perf_counter()

    LOGGER.info(
        "Generating %s from %d template rows; target size %s bytes",
        output_path,
        len(templates),
        f"{target_bytes:,}",
    )
    try:
        with temporary_path.open("wb", buffering=1024 * 1024) as output:
            output.write(header)
            while True:
                row_count += 1
                template = templates[(row_count - 1) % len(templates)]
                row = generated_row(template, row_count, rng)
                encoded = encode_csv_row([row.get(column, "") for column in fieldnames])
                if bytes_written + len(encoded) > target_bytes:
                    row_count -= 1
                    break
                output.write(encoded)
                bytes_written += len(encoded)
                if bytes_written >= next_progress:
                    LOGGER.info(
                        "Progress: %.1f%% (%s rows, %.2f MB)",
                        bytes_written / target_bytes * 100,
                        f"{row_count:,}",
                        bytes_written / 1_000_000,
                    )
                    next_progress += 100_000_000
        os.replace(temporary_path, output_path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    LOGGER.info(
        "Generated %s rows and %s bytes (%.6f GB) in %.2f seconds",
        f"{row_count:,}",
        f"{bytes_written:,}",
        bytes_written / 1_000_000_000,
        time.perf_counter() - started,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-bytes", type=int, default=DEFAULT_TARGET_BYTES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(args.template, args.output, args.target_bytes, args.seed)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    try:
        main()
    except Exception:
        LOGGER.exception("Large input generation failed")
        raise SystemExit(1)
