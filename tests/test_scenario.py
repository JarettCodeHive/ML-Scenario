import pandas as pd

from cluster_c.scenario import annual_coverage, breakeven_share, calculate_scenario, sensitivity, summarise


def sample() -> pd.DataFrame:
    return pd.DataFrame({
        "SUBSCRIPTION_REFERENCE_ID": ["A", "B"],
        "bookings_1y": [100, 100], "bookings_3y": [300, 300], "bookings_5y": [550, 550],
        "bookings_1y_p10": [90, 90], "bookings_3y_p10": [270, 270], "bookings_5y_p10": [500, 500],
        "bookings_1y_p90": [110, 110], "bookings_3y_p90": [330, 330], "bookings_5y_p90": [600, 600],
    })


def test_flexible_mix_and_coverage_are_transparent() -> None:
    details = calculate_scenario(sample(), 100, 50, 0)
    coverage = annual_coverage(details, 100, 50)
    summary = summarise(details, coverage)
    assert summary["baseline_commitment"] == 600
    assert summary["scenario_commitment"] == 650
    assert summary["average_term_years"] == 3
    assert summary["scenario_renewal_points"] == 6
    assert len(coverage) == 5


def test_sensitivity_and_breakeven() -> None:
    frame = sensitivity(sample(), 100, 0)
    assert len(frame) == 21
    assert breakeven_share(frame) is not None
