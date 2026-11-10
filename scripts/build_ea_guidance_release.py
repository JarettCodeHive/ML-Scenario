"""Build and publish a validated EA guidance release bundle."""

from __future__ import annotations

import argparse
import os

from cluster_c.unstructured.release import build_guidance_release


def parse_expected_rule_ids(value: str) -> set[str]:
    normalized = value.strip()
    if normalized.casefold() == "none":
        return set()
    values = {item.strip() for item in normalized.split(",") if item.strip()}
    if not values:
        raise argparse.ArgumentTypeError("Use a comma-separated list or the explicit value 'none'")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--application-version",
        default=os.getenv("APPLICATION_VERSION"),
        required=os.getenv("APPLICATION_VERSION") is None,
    )
    parser.add_argument(
        "--expected-active-rule-ids",
        type=parse_expected_rule_ids,
        default=(
            parse_expected_rule_ids(os.environ["EXPECTED_GUIDANCE_ACTIVE_RULE_IDS"])
            if "EXPECTED_GUIDANCE_ACTIVE_RULE_IDS" in os.environ
            else None
        ),
        required="EXPECTED_GUIDANCE_ACTIVE_RULE_IDS" not in os.environ,
        help="Comma-separated expected active rule IDs, or 'none'",
    )
    parser.add_argument(
        "--warning-policy",
        choices=["report", "fail"],
        default=os.getenv("GUIDANCE_RELEASE_WARNING_POLICY", "fail"),
    )
    args = parser.parse_args()
    metadata = build_guidance_release(
        application_version=args.application_version,
        expected_active_rule_ids=args.expected_active_rule_ids,
        warning_policy=args.warning_policy,
    )
    print(
        "EA guidance release published: "
        f"application={metadata['application_version']}, "
        f"active_rules={metadata['active_rule_count']}, "
        f"warnings={metadata['quality_warning_count']}"
    )


if __name__ == "__main__":
    main()
