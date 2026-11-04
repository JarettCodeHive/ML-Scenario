"""Generate the deterministic synthetic EA guidance PDF used by the POC."""

from __future__ import annotations

import hashlib
from pathlib import Path


OUTPUT_PATH = Path("data/unstructured/input/guidance/pdf/ea_general_guidance_sample_v1.pdf")

PAGES = [
    [
        "SAMPLE EA THREE-LENS DECISION GUIDANCE",
        "Version: v1",
        "Effective date: 2026-11-04",
        "Review status: NOT APPROVED",
        "",
        "IMPORTANT DISCLAIMER",
        "This is synthetic guidance created only to test the EA modelling pipeline.",
        "It is not Cisco policy, approved methodology, accounting guidance, or advice.",
        "Production thresholds must be reviewed and replaced by authorized EA owners.",
        "",
        "PURPOSE",
        "Define transparent interpretation rules for the existing three decision lenses:",
        "1. Growth",
        "2. Customer value",
        "3. Simplicity with scale",
        "",
        "SCOPE",
        "These rules apply at portfolio level to the Cluster C term-flexibility scenario.",
        "They define interpretation and thresholds; they are not attached to individual deals.",
        "The CatBoost model remains responsible for bookings estimates under each term option.",
    ],
    [
        "SAMPLE RULE DEFINITIONS",
        "",
        "GROWTH-G-001 - Commitment direction",
        "Measure: commitment_change_pct",
        "Favorable: value greater than 0 percent.",
        "Neutral: value equal to 0 percent within the approved rounding tolerance.",
        "Attention: value less than 0 percent.",
        "Explanation: compares flexible-term commitment with the uniform 3-year reference.",
        "",
        "GROWTH-G-002 - NPV direction",
        "Measure: npv_change",
        "Favorable: value greater than or equal to 0.",
        "Attention: value less than 0.",
        "Explanation: uses the configured annual discount rate and straight-line proxy.",
        "",
        "CUSTOMER-CV-001 - Term choice",
        "Measure: flexible_adoption_pct",
        "Report the selected flexible-deal share; do not label a higher share as better.",
        "Explanation: term choice is a proxy until direct customer outcomes are available.",
        "",
        "CUSTOMER-CV-002 - Expected term",
        "Measure: average_term_years",
        "Report direction against 3 years without assigning positive or negative sentiment.",
    ],
    [
        "SAMPLE RULE DEFINITIONS - CONTINUED",
        "",
        "SIMPLICITY-S-001 - Renewal decision points",
        "Measure: scenario_renewal_points minus baseline_renewal_points",
        "Favorable: change less than or equal to 0.",
        "Attention: change greater than 0.",
        "Explanation: more decision points are treated as additional operational touchpoints.",
        "",
        "SIMPLICITY-S-002 - Coverage volatility",
        "Measure: scenario_coverage_cv_pct minus baseline_coverage_cv_pct",
        "Favorable: change less than or equal to 0 percentage points.",
        "Attention: change greater than 0 percentage points.",
        "Explanation: this is a stability proxy, not revenue recognition logic.",
        "",
        "GOVERNANCE REQUIREMENTS",
        "Every extracted rule must retain source ID, version, page number, and source hash.",
        "Unknown or ambiguous language must not be converted into a decision threshold.",
        "Conflicting guidance must be flagged for review instead of silently overridden.",
        "Only approved guidance may influence production lens status or explanatory text.",
        "A new source version must produce a new prepared-rule version and audit record.",
        "Model outputs are observational and must not be presented as causal effects.",
    ],
]


def _escape_pdf_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _content_stream(lines: list[str]) -> bytes:
    commands = ["BT", "/F1 10 Tf", "50 742 Td", "14 TL"]
    for line in lines:
        commands.append(f"({_escape_pdf_text(line)}) Tj")
        commands.append("T*")
    commands.append("ET")
    return ("\n".join(commands) + "\n").encode("ascii")


def build_pdf(pages: list[list[str]]) -> bytes:
    """Build a small text-only PDF using only the Python standard library."""
    page_count = len(pages)
    font_object = 3 + page_count * 2
    page_objects = [3 + index * 2 for index in range(page_count)]
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: (
            f"<< /Type /Pages /Count {page_count} /Kids "
            f"[{' '.join(f'{number} 0 R' for number in page_objects)}] >>"
        ).encode("ascii"),
        font_object: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }

    for index, lines in enumerate(pages):
        page_object = page_objects[index]
        stream_object = page_object + 1
        stream = _content_stream(lines)
        objects[page_object] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_object} 0 R >> >> "
            f"/Contents {stream_object} 0 R >>"
        ).encode("ascii")
        objects[stream_object] = (
            f"<< /Length {len(stream)} >>\nstream\n".encode("ascii")
            + stream
            + b"endstream"
        )

    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_number in range(1, font_object + 1):
        offsets.append(len(output))
        output.extend(f"{object_number} 0 obj\n".encode("ascii"))
        output.extend(objects[object_number])
        output.extend(b"\nendobj\n")

    xref_offset = len(output)
    output.extend(f"xref\n0 {font_object + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        (
            f"trailer\n<< /Size {font_object + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(output)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    content = build_pdf(PAGES)
    OUTPUT_PATH.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    print(f"Generated {OUTPUT_PATH} ({len(content):,} bytes)")
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    main()
