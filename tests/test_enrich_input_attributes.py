import csv
from pathlib import Path

from scripts.enrich_input_attributes import enrich


def test_enrichment_adds_non_null_attributes_deterministically(tmp_path: Path) -> None:
    path = tmp_path / "input.csv"
    path.write_text("ID,VALUE\n1,a\n2,b\n", encoding="utf-8")

    assert enrich(path, seed=42) == 2

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert all(row["CX_UPSELL_GROUP"] for row in rows)
    assert all(row["XCAT_FLG"] in {"Y", "N"} for row in rows)
    assert all(float(row["DV_ATTRIBUTION_CD"]) != 0 for row in rows)
