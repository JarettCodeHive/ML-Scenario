from pathlib import Path

import pandas as pd

from cluster_c.data_analysis import analyze_csv, render_markdown


def test_chunked_analysis_reports_exact_nulls_and_ranges(tmp_path: Path) -> None:
    path = tmp_path / "sample.csv"
    pd.DataFrame({
        "CATEGORY": ["A", "B", None, "null", "  "],
        "AMOUNT": [1.0, 2.0, None, 4.0, 5.0],
        "TERM_START_DATE": ["2026-01-01", "bad", None, "2026-01-04", "2026-01-05"],
    }).to_csv(path, index=False)

    result = analyze_csv(path, chunk_size=2, unique_limit=10, top_value_limit=3)

    assert result.row_count == 5
    assert result.profiles["CATEGORY"].missing_count == 3
    assert result.profiles["AMOUNT"].missing_count == 1
    assert result.profiles["AMOUNT"].numeric_min == 1
    assert result.profiles["AMOUNT"].numeric_max == 5
    assert result.profiles["TERM_START_DATE"].missing_count == 1
    assert result.profiles["TERM_START_DATE"].date_count == 3
    assert result.profiles["TERM_START_DATE"].date_parse_failures == 1

    report = render_markdown(result, top_value_limit=3)
    assert "# Source Data Analysis Report" in report
    assert "## Null-value summary" in report
    assert "| `CATEGORY` | 3 | 60.00% | 2 |" in report
    assert "## Numeric summary" in report
    assert "## Date summary" in report


def test_high_cardinality_distinct_count_is_bounded(tmp_path: Path) -> None:
    path = tmp_path / "high-cardinality.csv"
    pd.DataFrame({"ID": [f"ID-{index}" for index in range(20)]}).to_csv(path, index=False)

    result = analyze_csv(path, chunk_size=5, unique_limit=10)

    profile = result.profiles["ID"]
    assert profile.unique_capped
    assert profile.distinct_display == ">10"
    assert "high cardinality" in profile.quality_flags(result.row_count)
