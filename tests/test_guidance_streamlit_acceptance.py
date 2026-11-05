from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


HARNESS = Path(__file__).parent / "fixtures" / "guidance" / "streamlit_acceptance_app.py"
TRACE_COLUMNS = {
    "lens",
    "status",
    "rule_id",
    "reason",
    "measure",
    "measure_value",
    "comparison_measure",
    "comparison_value",
    "operator",
    "threshold",
    "source_id",
    "source_version",
    "source_page",
    "source_fingerprint",
}


def _run() -> AppTest:
    app = AppTest.from_file(str(HARNESS), default_timeout=20).run()
    assert not app.exception
    return app


def _element(elements, label):
    return next(element for element in elements if element.label == label)


def _markdown(app: AppTest) -> list[str]:
    return [element.value for element in app.markdown]


def _trace(app: AppTest):
    return next(frame.value for frame in app.dataframe if "rule_id" in frame.value.columns)


def test_three_lens_cards_and_default_guidance_statuses_render() -> None:
    app = _run()
    markdown = _markdown(app)

    assert "#### 1. Growth" in markdown
    assert "#### 2. Customer value" in markdown
    assert "#### 3. Simplicity with scale" in markdown
    assert "**EA guidance: Favorable**" in markdown
    assert "**EA guidance: Informational**" in markdown
    assert "**EA guidance: No guidance rule applied**" in markdown
    assert _element(app.metric, "Structured scenario commitment").value == "$1,000,000"


def test_slider_threshold_changes_growth_lens_status() -> None:
    app = _run()

    _element(app.slider, "Commitment change percent").set_value(-1).run()

    assert not app.exception
    assert "**EA guidance: Attention required**" in _markdown(app)
    statuses = dict(zip(_trace(app).rule_id, _trace(app).status))
    assert statuses["ACCEPT-GLOBAL-GT"] == "attention"


def test_scoped_slider_threshold_changes_customer_value_status() -> None:
    app = _run()
    _element(app.multiselect, "SCMS").set_value(["ENTERPRISE"]).run()
    assert "**EA guidance: Favorable**" in _markdown(app)

    _element(app.slider, "Flexible adoption percent").set_value(90).run()

    statuses = dict(zip(_trace(app).rule_id, _trace(app).status))
    assert statuses["ACCEPT-SCMS-LT"] == "attention"
    assert "**EA guidance: Attention required**" in _markdown(app)


@pytest.mark.parametrize(
    ("filter_label", "selection", "expected_rule"),
    [
        ("SCMS", ["ENTERPRISE"], "ACCEPT-SCMS-LT"),
        ("Sales level 2", ["EMEA_NORTH"], "ACCEPT-SALES-LTE"),
        ("Suite", ["Cisco DNA Wireless"], "ACCEPT-SUITE-EQ"),
    ],
)
def test_exact_segment_filters_apply_the_expected_scoped_rule(filter_label, selection, expected_rule) -> None:
    app = _run()

    _element(app.multiselect, filter_label).set_value(selection).run()

    assert expected_rule in set(_trace(app).rule_id)


def test_unfiltered_and_mixed_filters_do_not_apply_segment_rule() -> None:
    app = _run()
    assert set(_trace(app).rule_id) == {
        "ACCEPT-GLOBAL-GT",
        "ACCEPT-GLOBAL-GTE",
        "ACCEPT-GLOBAL-REPORT",
    }

    _element(app.multiselect, "SCMS").set_value(["ENTERPRISE", "COMMERCIAL"]).run()

    assert "ACCEPT-SCMS-LT" not in set(_trace(app).rule_id)


def test_no_matching_scope_is_explicit_and_structured_result_remains_visible() -> None:
    app = _run()

    _element(app.multiselect, "Suite").set_value(["NO_MATCH"]).run()

    assert any("No rows match the selected scope" in element.value for element in app.info)
    assert _element(app.metric, "Structured scenario commitment").value == "$1,000,000"


def test_no_active_and_missing_guidance_do_not_hide_structured_result() -> None:
    app = _run()

    _element(app.selectbox, "Guidance state").set_value("No approved guidance").run()
    assert _markdown(app).count("**EA guidance: No guidance rule applied**") == 3
    assert any("No EA guidance rule" in element.value for element in app.info)
    assert _element(app.metric, "Structured scenario commitment").value == "$1,000,000"

    _element(app.selectbox, "Guidance state").set_value("Missing registry").run()
    assert any("EA guidance is unavailable" in element.value for element in app.warning)
    assert _element(app.metric, "Structured scenario commitment").value == "$1,000,000"


def test_traceability_contains_rule_decision_and_source_provenance() -> None:
    app = _run()
    _element(app.multiselect, "Sales level 2").set_value(["EMEA_NORTH"]).run()
    trace = _trace(app)

    assert set(trace.columns) == TRACE_COLUMNS
    sales_rule = trace.loc[trace.rule_id == "ACCEPT-SALES-LTE"].iloc[0]
    assert sales_rule.lens == "simplicity_with_scale"
    assert sales_rule.status == "favorable"
    assert sales_rule.reason
    assert sales_rule.measure == "scenario_renewal_points"
    assert sales_rule.measure_value == 9
    assert sales_rule.comparison_measure == "baseline_renewal_points"
    assert sales_rule.comparison_value == 10
    assert sales_rule.operator == "less_than_or_equal"
    assert sales_rule.threshold == 0
    assert sales_rule.source_id == "EA-TEST-APPROVED-001"
    assert sales_rule.source_version == "test-v1"
    assert sales_rule.source_page == 4
    assert sales_rule.source_fingerprint == "aaaaaaaaaaaa"
    assert any("generated 2026-11-04T12:00:00+00:00" in caption.value for caption in app.caption)
