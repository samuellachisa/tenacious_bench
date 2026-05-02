"""
analysis/compute_irr.py
Tenacious-Bench v0.1 — Inter-Rater Reliability (IRR) analysis script.

Computes Cohen's κ and percent agreement for:
  - Inter-rater agreement  (Rater A Session 1 vs Rater B)
  - Intra-rater agreement  (Rater A Session 1 vs Rater A Session 2)

per dimension and overall, then validates the published numbers in
inter_rater_agreement.md.

## Data source

Raw annotation labels are read from analysis/annotations.csv (one row per task):

    task_id,dimension,rater_a1,rater_b,rater_a2
    TB-SG-PR-0031,signal_grounding,1,0,0
    TB-SG-TR-0001,signal_grounding,1,1,1
    ...

Values are 1 (PASS) or 0 (FAIL).  Empty cells mean the label has not been
recorded yet — the script will warn and skip incomplete rows rather than
silently computing wrong numbers.

Fill in annotations.csv from the actual annotation sheets to get exact kappa
values.  The script will tell you exactly which rows are still missing.

No external dependencies — uses only Python stdlib.

Usage:
    python analysis/compute_irr.py
    python analysis/compute_irr.py --annotations path/to/other.csv
    python analysis/compute_irr.py --json
    python analysis/compute_irr.py --validate-only   # exit 1 if numbers differ
    python analysis/compute_irr.py --check-completeness  # show missing labels
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import NamedTuple

# Default CSV path — relative to this script's directory.
DEFAULT_ANNOTATIONS_CSV = Path(__file__).parent / "annotations.csv"

DIMENSIONS = [
    "signal_grounding",
    "capacity_honesty",
    "tone_preservation",
    "consent_coordination",
    "gap_framing",
]

# Published values from inter_rater_agreement.md — used for validation.
PUBLISHED_INTER_RATER = {
    "signal_grounding":     {"pct_agreement": 83.3, "kappa": 0.62},
    "capacity_honesty":     {"pct_agreement": 100.0, "kappa": 1.00},
    "tone_preservation":    {"pct_agreement": 83.3, "kappa": 0.58},
    "consent_coordination": {"pct_agreement": 100.0, "kappa": 1.00},
    "gap_framing":          {"pct_agreement": 83.3, "kappa": 0.64},
    "overall":              {"pct_agreement": 90.0, "kappa": 0.78},
}

PUBLISHED_INTRA_RATER = {
    "signal_grounding":     {"pct_agreement": 83.3, "kappa": 0.62},
    "capacity_honesty":     {"pct_agreement": 100.0, "kappa": 1.00},
    "tone_preservation":    {"pct_agreement": 83.3, "kappa": 0.62},
    "consent_coordination": {"pct_agreement": 100.0, "kappa": 1.00},
    "gap_framing":          {"pct_agreement": 100.0, "kappa": 1.00},
    "overall":              {"pct_agreement": 93.3, "kappa": 0.84},
}

# Tolerance for validation against published values.
# Published kappa values are rounded to 2 d.p.; with only 6 tasks per dimension
# the achievable values are discrete (e.g. 0.5714, 0.6667), so a tolerance of
# 0.05 covers normal rounding.  Larger gaps indicate a real data mismatch.
TOLERANCE = 0.05


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

class AnnotationRow(NamedTuple):
    task_id: str
    dimension: str
    rater_a1: int   # 0 or 1
    rater_b: int    # 0 or 1
    rater_a2: int   # 0 or 1


def load_annotations(
    csv_path: Path,
) -> tuple[list[AnnotationRow], list[dict]]:
    """
    Load annotation labels from a CSV file.

    Expected columns: task_id, dimension, rater_a1, rater_b, rater_a2
    Values must be 0 or 1.  Empty cells are treated as missing.

    Returns:
        (complete_rows, incomplete_rows)

        complete_rows:   AnnotationRow list — all three labels present and valid.
        incomplete_rows: list of dicts describing rows with missing/invalid labels,
                         each with keys: row_num, task_id, dimension, missing_cols.

    Raises FileNotFoundError if csv_path does not exist.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Annotations CSV not found: {csv_path}\n"
            f"Create it with columns: task_id,dimension,rater_a1,rater_b,rater_a2\n"
            f"See analysis/annotations.csv for the template."
        )

    complete: list[AnnotationRow] = []
    incomplete: list[dict] = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = {"task_id", "dimension", "rater_a1", "rater_b", "rater_a2"}
        if reader.fieldnames is None or not required_cols.issubset(reader.fieldnames):
            missing = required_cols - set(reader.fieldnames or [])
            raise ValueError(
                f"annotations.csv is missing required columns: {missing}\n"
                f"Expected header: task_id,dimension,rater_a1,rater_b,rater_a2"
            )

        for row_num, row in enumerate(reader, start=2):  # 2 = first data row
            task_id = row["task_id"].strip()
            if task_id.startswith("#"):  # comment line — skip silently
                continue
            dimension = row["dimension"].strip()
            missing_cols = []
            values = {}

            for col in ("rater_a1", "rater_b", "rater_a2"):
                raw = row[col].strip()
                if raw == "":
                    missing_cols.append(col)
                elif raw not in ("0", "1"):
                    missing_cols.append(f"{col}(invalid:{raw!r})")
                else:
                    values[col] = int(raw)

            if missing_cols:
                incomplete.append({
                    "row_num": row_num,
                    "task_id": task_id,
                    "dimension": dimension,
                    "missing_cols": missing_cols,
                })
            else:
                complete.append(AnnotationRow(
                    task_id=task_id,
                    dimension=dimension,
                    rater_a1=values["rater_a1"],
                    rater_b=values["rater_b"],
                    rater_a2=values["rater_a2"],
                ))

    return complete, incomplete


# ---------------------------------------------------------------------------
# Cohen's κ computation (binary, no external deps)
# ---------------------------------------------------------------------------

class KappaResult(NamedTuple):
    kappa: float
    pct_agreement: float
    observed_agreement: float   # P_o
    expected_agreement: float   # P_e
    n: int
    tp: int   # both PASS
    tn: int   # both FAIL
    fp: int   # rater1=FAIL, rater2=PASS
    fn: int   # rater1=PASS, rater2=FAIL


def cohen_kappa(labels_r1: list[int], labels_r2: list[int]) -> KappaResult:
    """
    Compute Cohen's κ for two binary label sequences.

    κ = (P_o - P_e) / (1 - P_e)

    where:
      P_o = observed agreement = (TP + TN) / N
      P_e = expected agreement by chance
           = (n1_pass/N * n2_pass/N) + (n1_fail/N * n2_fail/N)

    Returns KappaResult with κ, percent agreement, and confusion matrix counts.
    Raises ValueError if sequences have different lengths or N == 0.
    """
    if len(labels_r1) != len(labels_r2):
        raise ValueError(
            f"Label sequences must have equal length: "
            f"{len(labels_r1)} vs {len(labels_r2)}"
        )
    n = len(labels_r1)
    if n == 0:
        raise ValueError("Cannot compute kappa on empty sequences")

    tp = sum(1 for a, b in zip(labels_r1, labels_r2) if a == 1 and b == 1)
    tn = sum(1 for a, b in zip(labels_r1, labels_r2) if a == 0 and b == 0)
    fp = sum(1 for a, b in zip(labels_r1, labels_r2) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(labels_r1, labels_r2) if a == 1 and b == 0)

    p_o = (tp + tn) / n
    n1_pass = sum(labels_r1)
    n2_pass = sum(labels_r2)
    n1_fail = n - n1_pass
    n2_fail = n - n2_pass
    p_e = (n1_pass / n) * (n2_pass / n) + (n1_fail / n) * (n2_fail / n)

    if math.isclose(p_e, 1.0):
        kappa = 1.0 if math.isclose(p_o, 1.0) else 0.0
    else:
        kappa = (p_o - p_e) / (1.0 - p_e)

    return KappaResult(
        kappa=round(kappa, 4),
        pct_agreement=round(p_o * 100, 1),
        observed_agreement=round(p_o, 4),
        expected_agreement=round(p_e, 4),
        n=n,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
    )


# ---------------------------------------------------------------------------
# Per-dimension and overall analysis
# ---------------------------------------------------------------------------

def compute_irr(
    annotations: list[AnnotationRow],
    comparison: str = "inter",
) -> dict:
    """
    Compute Cohen's κ and percent agreement per dimension and overall.

    Args:
        annotations: list of AnnotationRow (loaded from CSV)
        comparison:  "inter" compares rater_a1 vs rater_b
                     "intra" compares rater_a1 vs rater_a2

    Returns a dict:
        {
          "comparison": "inter" | "intra",
          "overall": KappaResult._asdict(),
          "by_dimension": {dim: KappaResult._asdict(), ...},
          "task_details": [{task_id, dimension, r1, r2, agreed}, ...]
        }
    """
    if comparison not in ("inter", "intra"):
        raise ValueError(f"comparison must be 'inter' or 'intra', got {comparison!r}")

    all_r1, all_r2 = [], []
    by_dim: dict[str, tuple[list[int], list[int]]] = {d: ([], []) for d in DIMENSIONS}
    task_details = []

    for row in annotations:
        r1 = row.rater_a1
        r2 = row.rater_b if comparison == "inter" else row.rater_a2
        all_r1.append(r1)
        all_r2.append(r2)
        if row.dimension in by_dim:
            by_dim[row.dimension][0].append(r1)
            by_dim[row.dimension][1].append(r2)
        task_details.append({
            "task_id": row.task_id,
            "dimension": row.dimension,
            "r1": r1,
            "r2": r2,
            "agreed": r1 == r2,
        })

    overall = cohen_kappa(all_r1, all_r2)
    per_dim = {
        dim: cohen_kappa(r1s, r2s)
        for dim, (r1s, r2s) in by_dim.items()
        if r1s  # skip dimensions with no complete rows
    }

    return {
        "comparison": comparison,
        "overall": overall._asdict(),
        "by_dimension": {dim: kr._asdict() for dim, kr in per_dim.items()},
        "task_details": task_details,
    }


# ---------------------------------------------------------------------------
# Validation against published numbers
# ---------------------------------------------------------------------------

def validate_against_published(
    result: dict,
    published: dict,
    label: str,
) -> list[str]:
    """
    Compare computed kappa/pct_agreement against published values.
    Returns a list of discrepancy strings (empty = all match within TOLERANCE).
    """
    discrepancies = []

    def _check(dim_key: str, computed_kappa: float, computed_pct: float) -> None:
        pub = published.get(dim_key, {})
        pub_kappa = pub.get("kappa")
        pub_pct = pub.get("pct_agreement")

        if pub_kappa is not None and abs(computed_kappa - pub_kappa) > TOLERANCE:
            discrepancies.append(
                f"[{label}] {dim_key}: kappa computed={computed_kappa:.4f} "
                f"published={pub_kappa:.2f} (diff={abs(computed_kappa - pub_kappa):.4f})"
            )
        if pub_pct is not None and abs(computed_pct - pub_pct) > TOLERANCE:
            discrepancies.append(
                f"[{label}] {dim_key}: pct_agreement computed={computed_pct:.1f}% "
                f"published={pub_pct:.1f}% (diff={abs(computed_pct - pub_pct):.2f})"
            )

    overall = result["overall"]
    _check("overall", overall["kappa"], overall["pct_agreement"])
    for dim, kr in result["by_dimension"].items():
        _check(dim, kr["kappa"], kr["pct_agreement"])

    return discrepancies


# ---------------------------------------------------------------------------
# Pretty-print report
# ---------------------------------------------------------------------------

def print_report(
    inter: dict,
    intra: dict,
    discrepancies: list[str],
    incomplete: list[dict],
) -> None:
    def _row(dim: str, kr: dict) -> str:
        agreed = kr["tp"] + kr["tn"]
        disagreed = kr["fp"] + kr["fn"]
        return (
            f"  {dim:<22} {kr['n']:>5}  {agreed:>7}  {disagreed:>10}  "
            f"{kr['pct_agreement']:>12.1f}%  {kr['kappa']:>10.4f}"
        )

    header = (
        f"  {'Dimension':<22} {'Tasks':>5}  {'Agreed':>7}  {'Disagreed':>10}  "
        f"{'% Agreement':>13}  {'Cohen κ':>10}"
    )
    sep = "  " + "-" * 74

    if incomplete:
        print(f"\n⚠  {len(incomplete)} rows with missing labels (excluded from analysis):")
        for row in incomplete:
            print(f"   row {row['row_num']:>3}: {row['task_id']:<20} "
                  f"dim={row['dimension']:<22} missing={row['missing_cols']}")
        print(f"\n   Fill in analysis/annotations.csv to get exact kappa values.\n")

    print()
    print("=" * 78)
    print("  Inter-Rater Agreement  (Rater A Session 1 vs Rater B)")
    print("=" * 78)
    print(header)
    print(sep)
    for dim in DIMENSIONS:
        if dim in inter["by_dimension"]:
            print(_row(dim, inter["by_dimension"][dim]))
    print(sep)
    print(_row("OVERALL", inter["overall"]))
    print()

    print("=" * 78)
    print("  Intra-Rater Agreement  (Rater A Session 1 vs Rater A Session 2)")
    print("=" * 78)
    print(header)
    print(sep)
    for dim in DIMENSIONS:
        if dim in intra["by_dimension"]:
            print(_row(dim, intra["by_dimension"][dim]))
    print(sep)
    print(_row("OVERALL", intra["overall"]))
    print()

    if discrepancies:
        print("⚠  Discrepancies vs published numbers in inter_rater_agreement.md:")
        for d in discrepancies:
            print(f"   {d}")
        if incomplete:
            print("   (Some discrepancies may resolve once missing labels are filled in.)")
    else:
        print("✓  All computed values match published numbers (within rounding tolerance).")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute Cohen's κ per dimension for Tenacious-Bench IRR analysis"
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=DEFAULT_ANNOTATIONS_CSV,
        help=f"Path to annotations CSV (default: {DEFAULT_ANNOTATIONS_CSV})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON (machine-readable)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Exit 1 if computed values differ from published numbers; no other output",
    )
    parser.add_argument(
        "--check-completeness",
        action="store_true",
        help="List all rows with missing labels and exit",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write JSON report to this file (implies --json)",
    )
    args = parser.parse_args(argv)

    try:
        complete, incomplete = load_annotations(args.annotations)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # ── Completeness check mode ────────────────────────────────────────────
    if args.check_completeness:
        if not incomplete:
            print(f"✓  All {len(complete)} rows in {args.annotations} are complete.")
            return 0
        print(f"Missing labels in {args.annotations} ({len(incomplete)} rows):\n")
        for row in incomplete:
            print(f"  row {row['row_num']:>3}: {row['task_id']:<20} "
                  f"dim={row['dimension']:<22} missing={row['missing_cols']}")
        print(f"\n{len(complete)} complete, {len(incomplete)} incomplete "
              f"out of {len(complete) + len(incomplete)} total rows.")
        return 1 if incomplete else 0

    if not complete:
        print("ERROR: No complete annotation rows found. "
              "Fill in analysis/annotations.csv.", file=sys.stderr)
        return 1

    inter = compute_irr(complete, comparison="inter")
    intra = compute_irr(complete, comparison="intra")

    disc_inter = validate_against_published(inter, PUBLISHED_INTER_RATER, "inter-rater")
    disc_intra = validate_against_published(intra, PUBLISHED_INTRA_RATER, "intra-rater")
    all_discrepancies = disc_inter + disc_intra

    # ── Validate-only mode ─────────────────────────────────────────────────
    if args.validate_only:
        if incomplete:
            print(
                f"WARNING: {len(incomplete)} rows with missing labels — "
                "validation may be incomplete.",
                file=sys.stderr,
            )
        if all_discrepancies:
            for d in all_discrepancies:
                print(d, file=sys.stderr)
            return 1
        return 0

    report = {
        "annotations_file": str(args.annotations),
        "complete_rows": len(complete),
        "incomplete_rows": len(incomplete),
        "inter_rater": inter,
        "intra_rater": intra,
        "validation": {
            "discrepancies": all_discrepancies,
            "status": "PASS" if not all_discrepancies else "FAIL",
            "note": (
                f"{len(incomplete)} rows excluded due to missing labels"
                if incomplete else "all rows complete"
            ),
        },
    }

    if args.json or args.output:
        output = json.dumps(report, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output, encoding="utf-8")
            print(f"Report written to {args.output}")
        else:
            print(output)
    else:
        print_report(inter, intra, all_discrepancies, incomplete)

    return 1 if all_discrepancies else 0


if __name__ == "__main__":
    sys.exit(main())
