"""Memory-bounded CSV profiling used for structured source-data analysis."""

from __future__ import annotations

import math
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_NULL_LITERALS = ["null", "NULL", "None", "<NULL>"]
LOGGER = logging.getLogger(__name__)


def _markdown(value: Any) -> str:
    text = str(value).replace("\n", " ").replace("\r", " ").replace("|", "\\|")
    return text if len(text) <= 80 else text[:77] + "..."


def _human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:,.2f} {unit}"
        value /= 1024
    return f"{value:,.2f} TiB"


@dataclass
class ColumnProfile:
    name: str
    unique_limit: int
    top_value_limit: int
    missing_count: int = 0
    non_null_count: int = 0
    observed_dtypes: set[str] = field(default_factory=set)
    unique_values: set[str] = field(default_factory=set)
    unique_capped: bool = False
    value_counts: Counter[str] = field(default_factory=Counter)
    value_counts_exact: bool = True
    numeric_count: int = 0
    numeric_sum: float = 0.0
    numeric_sum_squares: float = 0.0
    numeric_min: float | None = None
    numeric_max: float | None = None
    date_count: int = 0
    date_parse_failures: int = 0
    date_min: pd.Timestamp | None = None
    date_max: pd.Timestamp | None = None

    def update(self, series: pd.Series) -> None:
        self.observed_dtypes.add(str(series.dtype))
        missing = series.isna()
        if pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype):
            missing |= series.astype("string").str.strip().eq("").fillna(False)
        self.missing_count += int(missing.sum())
        values = series.loc[~missing]
        self.non_null_count += len(values)
        if values.empty:
            return

        text_values = values.astype("string")
        if not self.unique_capped:
            remaining = self.unique_limit + 1 - len(self.unique_values)
            self.unique_values.update(text_values.drop_duplicates().head(remaining).tolist())
            if len(self.unique_values) > self.unique_limit:
                self.unique_capped = True
                self.unique_values = set(list(self.unique_values)[: self.unique_limit])

        counts = text_values.value_counts(dropna=False)
        if len(counts) <= self.unique_limit and len(self.value_counts) <= self.unique_limit:
            self.value_counts.update({str(value): int(count) for value, count in counts.items()})
            if len(self.value_counts) > self.unique_limit:
                self.value_counts_exact = False
                self.value_counts = Counter(dict(self.value_counts.most_common(max(100, self.top_value_limit * 20))))
        else:
            self.value_counts_exact = False
            candidates = counts.head(max(100, self.top_value_limit * 20))
            self.value_counts.update({str(value): int(count) for value, count in candidates.items()})
            self.value_counts = Counter(dict(self.value_counts.most_common(max(100, self.top_value_limit * 20))))

        if pd.api.types.is_numeric_dtype(values.dtype):
            numeric = pd.to_numeric(values, errors="coerce").astype("float64")
            numeric = numeric[numeric.map(math.isfinite)]
            if not numeric.empty:
                self.numeric_count += len(numeric)
                self.numeric_sum += float(numeric.sum())
                self.numeric_sum_squares += float((numeric * numeric).sum())
                chunk_min = float(numeric.min())
                chunk_max = float(numeric.max())
                self.numeric_min = chunk_min if self.numeric_min is None else min(self.numeric_min, chunk_min)
                self.numeric_max = chunk_max if self.numeric_max is None else max(self.numeric_max, chunk_max)

        if "DATE" in self.name.upper():
            parsed = pd.to_datetime(values, errors="coerce", utc=True)
            valid = parsed.dropna()
            self.date_count += len(valid)
            self.date_parse_failures += int(parsed.isna().sum())
            if not valid.empty:
                chunk_min = valid.min()
                chunk_max = valid.max()
                self.date_min = chunk_min if self.date_min is None else min(self.date_min, chunk_min)
                self.date_max = chunk_max if self.date_max is None else max(self.date_max, chunk_max)

    @property
    def inferred_dtype(self) -> str:
        return ", ".join(sorted(self.observed_dtypes)) or "unknown"

    @property
    def distinct_display(self) -> str:
        return f">{self.unique_limit:,}" if self.unique_capped else f"{len(self.unique_values):,}"

    def missing_pct(self, row_count: int) -> float:
        return self.missing_count * 100 / row_count if row_count else 0.0

    def quality_flags(self, row_count: int) -> list[str]:
        flags: list[str] = []
        pct = self.missing_pct(row_count)
        if self.missing_count == row_count and row_count:
            flags.append("all null")
        elif pct >= 50:
            flags.append("high null rate")
        elif self.missing_count:
            flags.append("contains nulls")
        if not self.unique_capped and len(self.unique_values) == 1 and self.non_null_count:
            flags.append("constant when populated")
        if self.unique_capped:
            flags.append("high cardinality")
        if self.date_parse_failures:
            flags.append("unparseable dates")
        return flags


@dataclass(frozen=True)
class DataAnalysisResult:
    input_path: Path
    file_size: int
    row_count: int
    columns: list[str]
    profiles: dict[str, ColumnProfile]
    generated_at_utc: datetime
    chunk_size: int

    @property
    def total_missing(self) -> int:
        return sum(profile.missing_count for profile in self.profiles.values())


def analyze_csv(
    input_path: Path,
    *,
    chunk_size: int = 100_000,
    unique_limit: int = 10_000,
    top_value_limit: int = 10,
) -> DataAnalysisResult:
    """Profile a CSV in chunks while keeping null counts exact."""
    input_path = input_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {input_path}")
    if input_path.suffix.casefold() != ".csv":
        raise ValueError(f"Input must be a CSV file: {input_path}")
    if chunk_size < 1 or unique_limit < 1 or top_value_limit < 1:
        raise ValueError("chunk_size, unique_limit, and top_value_limit must be positive")

    reader = pd.read_csv(
        input_path,
        chunksize=chunk_size,
        low_memory=False,
        keep_default_na=True,
        na_values=DEFAULT_NULL_LITERALS,
    )
    profiles: dict[str, ColumnProfile] = {}
    columns: list[str] = []
    row_count = 0
    for chunk_number, chunk in enumerate(reader, start=1):
        if not columns:
            columns = list(chunk.columns)
            profiles = {
                column: ColumnProfile(column, unique_limit, top_value_limit)
                for column in columns
            }
        elif list(chunk.columns) != columns:
            raise ValueError("CSV columns changed between chunks")
        row_count += len(chunk)
        for column in columns:
            profiles[column].update(chunk[column])
        if chunk_number == 1 or chunk_number % 10 == 0:
            LOGGER.info("Profiled %s rows", f"{row_count:,}")

    if not columns:
        header = pd.read_csv(input_path, nrows=0)
        columns = list(header.columns)
        profiles = {
            column: ColumnProfile(column, unique_limit, top_value_limit)
            for column in columns
        }
    return DataAnalysisResult(
        input_path=input_path,
        file_size=input_path.stat().st_size,
        row_count=row_count,
        columns=columns,
        profiles=profiles,
        generated_at_utc=datetime.now(timezone.utc),
        chunk_size=chunk_size,
    )


def render_markdown(result: DataAnalysisResult, top_value_limit: int = 10) -> str:
    """Render a self-contained Markdown data-analysis report."""
    total_cells = result.row_count * len(result.columns)
    missing_pct = result.total_missing * 100 / total_cells if total_cells else 0.0
    lines = [
        "# Source Data Analysis Report",
        "",
        "## Dataset summary",
        "",
        f"- Input file: `{result.input_path}`",
        f"- Generated at: `{result.generated_at_utc.isoformat()}`",
        f"- File size: **{_human_bytes(result.file_size)}**",
        f"- Rows: **{result.row_count:,}**",
        f"- Columns: **{len(result.columns):,}**",
        f"- Total cells: **{total_cells:,}**",
        f"- Null/blank cells: **{result.total_missing:,} ({missing_pct:.2f}%)**",
        f"- Processing chunk size: **{result.chunk_size:,} rows**",
        "",
        "## Null-value summary",
        "",
        "Null counts and percentages in this table are exact. Empty fields, whitespace-only fields, and the configured literals `null`, `NULL`, `None`, and `<NULL>` are treated as missing.",
        "",
        "| Column | Null count | Null % | Non-null count |",
        "|---|---:|---:|---:|",
    ]
    for profile in sorted(result.profiles.values(), key=lambda item: (-item.missing_count, result.columns.index(item.name))):
        lines.append(
            f"| `{_markdown(profile.name)}` | {profile.missing_count:,} | "
            f"{profile.missing_pct(result.row_count):.2f}% | {profile.non_null_count:,} |"
        )

    lines.extend([
        "",
        "## Column profile",
        "",
        "Distinct-value tracking is capped to keep the analysis safe for multi-million-row files.",
        "",
        "| Column | Observed CSV dtype(s) | Distinct values | Quality flags |",
        "|---|---|---:|---|",
    ])
    for column in result.columns:
        profile = result.profiles[column]
        flags = ", ".join(profile.quality_flags(result.row_count)) or "none"
        lines.append(
            f"| `{_markdown(column)}` | `{_markdown(profile.inferred_dtype)}` | "
            f"{profile.distinct_display} | {_markdown(flags)} |"
        )

    numeric = [profile for profile in result.profiles.values() if profile.numeric_count]
    lines.extend(["", "## Numeric summary", ""])
    if numeric:
        lines.extend([
            "| Column | Numeric count | Minimum | Mean | Maximum | Population std. dev. |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for profile in numeric:
            mean = profile.numeric_sum / profile.numeric_count
            variance = max(0.0, profile.numeric_sum_squares / profile.numeric_count - mean * mean)
            lines.append(
                f"| `{_markdown(profile.name)}` | {profile.numeric_count:,} | "
                f"{profile.numeric_min:,.6g} | {mean:,.6g} | {profile.numeric_max:,.6g} | "
                f"{math.sqrt(variance):,.6g} |"
            )
    else:
        lines.append("No columns were consistently read as numeric values.")

    date_profiles = [profile for profile in result.profiles.values() if "DATE" in profile.name.upper()]
    lines.extend(["", "## Date summary", ""])
    if date_profiles:
        lines.extend([
            "| Column | Parsed dates | Unparseable non-null values | Minimum | Maximum |",
            "|---|---:|---:|---|---|",
        ])
        for profile in date_profiles:
            minimum = profile.date_min.isoformat() if profile.date_min is not None else "N/A"
            maximum = profile.date_max.isoformat() if profile.date_max is not None else "N/A"
            lines.append(
                f"| `{_markdown(profile.name)}` | {profile.date_count:,} | "
                f"{profile.date_parse_failures:,} | `{minimum}` | `{maximum}` |"
            )
    else:
        lines.append("No columns with `DATE` in their names were found.")

    lines.extend(["", "## Frequent values for bounded-cardinality columns", ""])
    bounded = [profile for profile in result.profiles.values() if not profile.unique_capped and profile.non_null_count]
    if not bounded:
        lines.append("No bounded-cardinality populated columns were found.")
    for profile in bounded:
        accuracy = "exact" if profile.value_counts_exact else "candidate-based"
        lines.extend([
            f"### `{_markdown(profile.name)}`",
            "",
            f"Counts shown below are **{accuracy}**.",
            "",
            "| Value | Count | % of non-null |",
            "|---|---:|---:|",
        ])
        for value, count in profile.value_counts.most_common(top_value_limit):
            pct = count * 100 / profile.non_null_count if profile.non_null_count else 0.0
            lines.append(f"| `{_markdown(value)}` | {count:,} | {pct:.2f}% |")
        lines.append("")

    flagged = [profile for profile in result.profiles.values() if profile.quality_flags(result.row_count)]
    lines.extend(["## Data-quality observations", ""])
    if flagged:
        for profile in flagged:
            lines.append(f"- `{_markdown(profile.name)}`: {', '.join(profile.quality_flags(result.row_count))}.")
    else:
        lines.append("- No automatic column-level quality flags were raised.")

    lines.extend([
        "",
        "## Interpretation and limitations",
        "",
        "- Null counts, row counts, numeric summaries, and date ranges are calculated across the complete CSV.",
        "- Distinct counts are exact only when they do not exceed the configured cap; larger counts are displayed as greater than the cap.",
        "- Frequent-value counts are exact for bounded-cardinality columns unless explicitly labelled candidate-based.",
        "- CSV dtypes describe how pandas observed each chunk; they are not a replacement for the project data contract.",
        "- This report does not calculate full-row duplicate counts because doing so exactly can consume substantial memory for a 10-million-row file.",
        "- Identifier columns may be numeric in the source but should still be treated as identifiers, not continuous measures.",
        "- Review high-null, constant, high-cardinality, and unparseable-date columns before preparation or training.",
        "",
    ])
    return "\n".join(lines)
