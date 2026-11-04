import pytest

from cluster_c.unstructured.schemas import GuidanceRule, validate_rule_set


SOURCE_HASH = "c1c07722a93f19d9d0f5f886878a01c40f20a7d6dc00c86e1d9d69e33a0ea3da"


def rule_values(**overrides):
    values = {
        "rule_id": "GROWTH-G-001",
        "lens": "growth",
        "measure": "commitment_change_pct",
        "operator": "greater_than",
        "threshold": 0.0,
        "comparison_measure": None,
        "status_if_matched": "favorable",
        "status_if_not_matched": "attention",
        "explanation": "Positive commitment change is favorable in the sample guidance.",
        "scope_type": "global",
        "scope_column": None,
        "scope_value": None,
        "source_id": "EA-GUIDE-SAMPLE-001",
        "source_version": "v1",
        "source_page": 2,
        "source_sha256": SOURCE_HASH,
        "review_status": "not_approved",
    }
    values.update(overrides)
    return values


def test_controlled_guidance_rule_round_trip() -> None:
    rule = GuidanceRule.from_mapping(rule_values())

    assert rule.rule_id == "GROWTH-G-001"
    assert rule.to_dict() == rule_values()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("lens", "profitability", "lens must be"),
        ("measure", "arbitrary_python", "measure must be"),
        ("operator", "eval", "operator must be"),
        ("source_sha256", "bad-hash", "source_sha256"),
        ("review_status", "live", "review_status"),
    ],
)
def test_guidance_rule_rejects_uncontrolled_values(field, value, message) -> None:
    with pytest.raises(ValueError, match=message):
        GuidanceRule.from_mapping(rule_values(**{field: value}))


def test_portfolio_filter_scope_requires_an_allowed_column_and_value() -> None:
    valid = GuidanceRule.from_mapping(rule_values(
        rule_id="GROWTH-G-EMEA",
        scope_type="portfolio_filter",
        scope_column="SCMS",
        scope_value="ENTERPRISE",
    ))
    assert valid.scope_column == "SCMS"

    with pytest.raises(ValueError, match="scope_column"):
        GuidanceRule.from_mapping(rule_values(
            scope_type="portfolio_filter",
            scope_column="SUBSCRIPTION_REFERENCE_ID",
            scope_value="SR1",
        ))


def test_guidance_rule_set_rejects_duplicate_ids() -> None:
    first = GuidanceRule.from_mapping(rule_values())
    second = GuidanceRule.from_mapping(rule_values(explanation="Duplicate rule"))

    with pytest.raises(ValueError, match="duplicates"):
        validate_rule_set([first, second])
