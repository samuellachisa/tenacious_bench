"""
tests/test_irr.py
Unit tests for analysis/compute_irr.py.

Covers:
  - cohen_kappa() correctness on known inputs and edge cases
  - load_annotations() CSV parsing, missing-cell detection, invalid values
  - compute_irr() structure and inter-vs-intra distinction
  - validate_against_published() discrepancy detection
  - Integration: complete CSV → kappa matches published numbers
"""

import csv
import math
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.compute_irr import (
    AnnotationRow,
    DIMENSIONS,
    PUBLISHED_INTER_RATER,
    PUBLISHED_INTRA_RATER,
    TOLERANCE,
    cohen_kappa,
    compute_irr,
    load_annotations,
    validate_against_published,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(tmp_dir: Path, rows: list[dict]) -> Path:
    """Write a CSV with the standard annotation columns."""
    path = tmp_dir / "annotations.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["task_id", "dimension", "rater_a1", "rater_b", "rater_a2"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def _make_row(
    task_id: str = "TB-TEST-0001",
    dimension: str = "signal_grounding",
    a1: int | str = 1,
    b: int | str = 1,
    a2: int | str = 1,
) -> dict:
    return {
        "task_id": task_id,
        "dimension": dimension,
        "rater_a1": a1,
        "rater_b": b,
        "rater_a2": a2,
    }


# ---------------------------------------------------------------------------
# cohen_kappa() unit tests
# ---------------------------------------------------------------------------

class TestCohenKappa:

    def test_perfect_agreement_pass(self):
        r = cohen_kappa([1, 1, 1, 1], [1, 1, 1, 1])
        assert r.kappa == 1.0
        assert r.pct_agreement == 100.0

    def test_perfect_agreement_fail(self):
        r = cohen_kappa([0, 0, 0, 0], [0, 0, 0, 0])
        assert r.kappa == 1.0
        assert r.pct_agreement == 100.0

    def test_perfect_agreement_mixed(self):
        labels = [1, 0, 1, 0, 1, 1]
        r = cohen_kappa(labels, labels)
        assert r.kappa == 1.0
        assert r.pct_agreement == 100.0

    def test_zero_agreement(self):
        r = cohen_kappa([1, 1, 0, 0], [0, 0, 1, 1])
        assert r.kappa < 0
        assert r.pct_agreement == 0.0

    def test_chance_agreement(self):
        # P_o = 0.5, P_e = 0.5 → κ = 0
        r = cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0])
        assert math.isclose(r.kappa, 0.0, abs_tol=1e-9)

    def test_known_values(self):
        """
        Hand-computed:
          r1=[1,1,0,0,1,0], r2=[1,0,0,0,1,1]
          TP=2, TN=2, FP=1, FN=1
          P_o=4/6, P_e=0.5  →  κ=1/3
        """
        r = cohen_kappa([1, 1, 0, 0, 1, 0], [1, 0, 0, 0, 1, 1])
        assert math.isclose(r.kappa, 1 / 3, abs_tol=1e-4)
        assert r.tp == 2
        assert r.tn == 2
        assert r.fp == 1
        assert r.fn == 1

    def test_confusion_matrix_counts(self):
        r = cohen_kappa([1, 0, 1, 0], [1, 1, 0, 0])
        assert r.tp == 1
        assert r.tn == 1
        assert r.fp == 1
        assert r.fn == 1
        assert r.n == 4

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            cohen_kappa([1, 0], [1, 0, 1])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            cohen_kappa([], [])

    def test_single_item_agreement(self):
        r = cohen_kappa([1], [1])
        assert r.kappa == 1.0

    def test_single_item_disagreement(self):
        r = cohen_kappa([1], [0])
        assert r.pct_agreement == 0.0


# ---------------------------------------------------------------------------
# load_annotations() tests
# ---------------------------------------------------------------------------

class TestLoadAnnotations:

    def test_loads_complete_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(Path(tmp), [
                _make_row("TB-SG-0001", "signal_grounding", 1, 1, 1),
                _make_row("TB-SG-0002", "signal_grounding", 0, 0, 0),
            ])
            complete, incomplete = load_annotations(path)
        assert len(complete) == 2
        assert incomplete == []
        assert complete[0] == AnnotationRow("TB-SG-0001", "signal_grounding", 1, 1, 1)

    def test_detects_missing_rater_a1(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(Path(tmp), [
                _make_row("TB-SG-0001", "signal_grounding", "", 1, 1),
            ])
            complete, incomplete = load_annotations(path)
        assert len(complete) == 0
        assert len(incomplete) == 1
        assert "rater_a1" in incomplete[0]["missing_cols"]

    def test_detects_missing_rater_b(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(Path(tmp), [
                _make_row("TB-SG-0001", "signal_grounding", 1, "", 1),
            ])
            complete, incomplete = load_annotations(path)
        assert "rater_b" in incomplete[0]["missing_cols"]

    def test_detects_missing_rater_a2(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(Path(tmp), [
                _make_row("TB-SG-0001", "signal_grounding", 1, 1, ""),
            ])
            complete, incomplete = load_annotations(path)
        assert "rater_a2" in incomplete[0]["missing_cols"]

    def test_detects_all_three_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(Path(tmp), [
                _make_row("TB-SG-0001", "signal_grounding", "", "", ""),
            ])
            complete, incomplete = load_annotations(path)
        assert len(incomplete[0]["missing_cols"]) == 3

    def test_detects_invalid_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(Path(tmp), [
                _make_row("TB-SG-0001", "signal_grounding", 2, 1, 1),  # 2 is invalid
            ])
            complete, incomplete = load_annotations(path)
        assert len(complete) == 0
        assert len(incomplete) == 1
        assert any("rater_a1" in col for col in incomplete[0]["missing_cols"])

    def test_mixed_complete_and_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(Path(tmp), [
                _make_row("TB-SG-0001", "signal_grounding", 1, 1, 1),   # complete
                _make_row("TB-SG-0002", "signal_grounding", 1, "", 0),  # incomplete
                _make_row("TB-SG-0003", "signal_grounding", 0, 0, 0),   # complete
            ])
            complete, incomplete = load_annotations(path)
        assert len(complete) == 2
        assert len(incomplete) == 1
        assert incomplete[0]["task_id"] == "TB-SG-0002"

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_annotations(Path("/nonexistent/path/annotations.csv"))

    def test_missing_column_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = tmp + "/bad.csv"
            with open(path, "w") as f:
                f.write("task_id,dimension,rater_a1\n")  # missing rater_b, rater_a2
                f.write("TB-SG-0001,signal_grounding,1\n")
            with pytest.raises(ValueError, match="missing required columns"):
                load_annotations(Path(path))

    def test_row_numbers_in_incomplete(self):
        """Row numbers should be 1-indexed from the data (row 2 = first data row)."""
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(Path(tmp), [
                _make_row("TB-SG-0001", "signal_grounding", 1, 1, 1),
                _make_row("TB-SG-0002", "signal_grounding", "", 1, 1),
            ])
            _, incomplete = load_annotations(path)
        assert incomplete[0]["row_num"] == 3  # header=1, first data=2, second data=3


# ---------------------------------------------------------------------------
# compute_irr() tests
# ---------------------------------------------------------------------------

class TestComputeIrr:

    def _make_annotations(self, n: int = 6) -> list[AnnotationRow]:
        """Make n complete annotation rows, all agreeing."""
        return [
            AnnotationRow(f"TB-SG-{i:04d}", "signal_grounding", 1, 1, 1)
            for i in range(n)
        ]

    def test_returns_expected_keys(self):
        result = compute_irr(self._make_annotations(), comparison="inter")
        assert "comparison" in result
        assert "overall" in result
        assert "by_dimension" in result
        assert "task_details" in result

    def test_task_details_length(self):
        rows = self._make_annotations(6)
        result = compute_irr(rows, comparison="inter")
        assert len(result["task_details"]) == 6

    def test_task_details_agreed_field(self):
        rows = [
            AnnotationRow("TB-SG-0001", "signal_grounding", 1, 1, 1),
            AnnotationRow("TB-SG-0002", "signal_grounding", 1, 0, 1),  # inter disagree
        ]
        result = compute_irr(rows, comparison="inter")
        details = {td["task_id"]: td for td in result["task_details"]}
        assert details["TB-SG-0001"]["agreed"] is True
        assert details["TB-SG-0002"]["agreed"] is False

    def test_inter_uses_rater_b(self):
        """Inter comparison must use rater_b, not rater_a2."""
        rows = [
            AnnotationRow("TB-SG-0001", "signal_grounding", 1, 0, 1),  # A1≠B, A1==A2
        ]
        inter = compute_irr(rows, comparison="inter")
        intra = compute_irr(rows, comparison="intra")
        assert inter["task_details"][0]["agreed"] is False
        assert intra["task_details"][0]["agreed"] is True

    def test_intra_uses_rater_a2(self):
        """Intra comparison must use rater_a2, not rater_b."""
        rows = [
            AnnotationRow("TB-SG-0001", "signal_grounding", 1, 1, 0),  # A1==B, A1≠A2
        ]
        inter = compute_irr(rows, comparison="inter")
        intra = compute_irr(rows, comparison="intra")
        assert inter["task_details"][0]["agreed"] is True
        assert intra["task_details"][0]["agreed"] is False

    def test_invalid_comparison_raises(self):
        with pytest.raises(ValueError, match="comparison"):
            compute_irr(self._make_annotations(), comparison="pairwise")

    def test_skips_unknown_dimensions(self):
        """Rows with dimensions not in DIMENSIONS are counted in overall but not by_dimension."""
        rows = [
            AnnotationRow("TB-XX-0001", "unknown_dim", 1, 1, 1),
            AnnotationRow("TB-SG-0001", "signal_grounding", 1, 1, 1),
        ]
        result = compute_irr(rows, comparison="inter")
        assert result["overall"]["n"] == 2
        assert "unknown_dim" not in result["by_dimension"]
        assert "signal_grounding" in result["by_dimension"]


# ---------------------------------------------------------------------------
# validate_against_published() tests
# ---------------------------------------------------------------------------

class TestValidateAgainstPublished:

    def _perfect_result(self) -> dict:
        """Build a fake result dict that exactly matches PUBLISHED_INTER_RATER."""
        by_dim = {
            dim: {"kappa": PUBLISHED_INTER_RATER[dim]["kappa"],
                  "pct_agreement": PUBLISHED_INTER_RATER[dim]["pct_agreement"]}
            for dim in DIMENSIONS
        }
        return {
            "overall": {
                "kappa": PUBLISHED_INTER_RATER["overall"]["kappa"],
                "pct_agreement": PUBLISHED_INTER_RATER["overall"]["pct_agreement"],
            },
            "by_dimension": by_dim,
        }

    def test_no_discrepancies_on_exact_match(self):
        result = self._perfect_result()
        disc = validate_against_published(result, PUBLISHED_INTER_RATER, "inter-rater")
        assert disc == []

    def test_detects_kappa_discrepancy(self):
        result = self._perfect_result()
        result["by_dimension"]["signal_grounding"]["kappa"] = 0.99  # wrong
        disc = validate_against_published(result, PUBLISHED_INTER_RATER, "inter-rater")
        assert any("signal_grounding" in d and "kappa" in d for d in disc)

    def test_detects_pct_discrepancy(self):
        result = self._perfect_result()
        result["overall"]["pct_agreement"] = 50.0  # wrong
        disc = validate_against_published(result, PUBLISHED_INTER_RATER, "inter-rater")
        assert any("overall" in d and "pct_agreement" in d for d in disc)

    def test_within_tolerance_not_flagged(self):
        result = self._perfect_result()
        # Nudge by less than TOLERANCE
        result["by_dimension"]["capacity_honesty"]["kappa"] = 1.00 + (TOLERANCE / 2)
        disc = validate_against_published(result, PUBLISHED_INTER_RATER, "inter-rater")
        assert not any("capacity_honesty" in d for d in disc)

    def test_exactly_at_tolerance_is_flagged(self):
        """The condition is abs(diff) > TOLERANCE, so exactly at TOLERANCE is flagged."""
        result = self._perfect_result()
        result["by_dimension"]["capacity_honesty"]["kappa"] = 1.00 + TOLERANCE + 0.001
        disc = validate_against_published(result, PUBLISHED_INTER_RATER, "inter-rater")
        assert any("capacity_honesty" in d for d in disc)


# ---------------------------------------------------------------------------
# Integration: CSV → kappa
# ---------------------------------------------------------------------------

class TestCsvIntegration:
    """End-to-end: write a CSV, load it, compute kappa, check the result."""

    def _full_csv(self, tmp: Path) -> Path:
        """
        Write a complete 30-row CSV with known labels.
        Disagreements match the document:
          Inter: TB-SG-PR-0031 (A1=1,B=0), TB-TP-PR-0018 (A1=0,B=1), TB-GF-ML-0007 (A1=1,B=0)
          Intra: TB-SG-PR-0031 (A1=1,A2=0), TB-TP-ML-0023 (A1=1,A2=0)
        All other tasks agree across all three raters.
        """
        rows = [
            # signal_grounding — 1 inter disagree, 1 intra disagree (same task)
            {"task_id": "TB-SG-PR-0031", "dimension": "signal_grounding", "rater_a1": 1, "rater_b": 0, "rater_a2": 0},
            {"task_id": "TB-SG-TR-0001", "dimension": "signal_grounding", "rater_a1": 1, "rater_b": 1, "rater_a2": 1},
            {"task_id": "TB-SG-TR-0002", "dimension": "signal_grounding", "rater_a1": 0, "rater_b": 0, "rater_a2": 0},
            {"task_id": "TB-SG-TR-0003", "dimension": "signal_grounding", "rater_a1": 0, "rater_b": 0, "rater_a2": 0},
            {"task_id": "TB-SG-TR-0004", "dimension": "signal_grounding", "rater_a1": 0, "rater_b": 0, "rater_a2": 0},
            {"task_id": "TB-SG-TR-0005", "dimension": "signal_grounding", "rater_a1": 0, "rater_b": 0, "rater_a2": 0},
            # capacity_honesty — all agree
            {"task_id": "TB-CH-TR-0001", "dimension": "capacity_honesty", "rater_a1": 1, "rater_b": 1, "rater_a2": 1},
            {"task_id": "TB-CH-TR-0002", "dimension": "capacity_honesty", "rater_a1": 1, "rater_b": 1, "rater_a2": 1},
            {"task_id": "TB-CH-TR-0003", "dimension": "capacity_honesty", "rater_a1": 0, "rater_b": 0, "rater_a2": 0},
            {"task_id": "TB-CH-TR-0004", "dimension": "capacity_honesty", "rater_a1": 0, "rater_b": 0, "rater_a2": 0},
            {"task_id": "TB-CH-TR-0005", "dimension": "capacity_honesty", "rater_a1": 1, "rater_b": 1, "rater_a2": 1},
            {"task_id": "TB-CH-TR-0006", "dimension": "capacity_honesty", "rater_a1": 0, "rater_b": 0, "rater_a2": 0},
            # tone_preservation — 1 inter disagree (TB-TP-PR-0018), 1 intra disagree (TB-TP-ML-0023)
            {"task_id": "TB-TP-PR-0018", "dimension": "tone_preservation", "rater_a1": 0, "rater_b": 1, "rater_a2": 0},
            {"task_id": "TB-TP-ML-0023", "dimension": "tone_preservation", "rater_a1": 1, "rater_b": 1, "rater_a2": 0},
            {"task_id": "TB-TP-TR-0001", "dimension": "tone_preservation", "rater_a1": 0, "rater_b": 0, "rater_a2": 0},
            {"task_id": "TB-TP-TR-0002", "dimension": "tone_preservation", "rater_a1": 0, "rater_b": 0, "rater_a2": 0},
            {"task_id": "TB-TP-TR-0003", "dimension": "tone_preservation", "rater_a1": 0, "rater_b": 0, "rater_a2": 0},
            {"task_id": "TB-TP-TR-0004", "dimension": "tone_preservation", "rater_a1": 0, "rater_b": 0, "rater_a2": 0},
            # consent_coordination — all agree
            {"task_id": "TB-CC-TR-0001", "dimension": "consent_coordination", "rater_a1": 1, "rater_b": 1, "rater_a2": 1},
            {"task_id": "TB-CC-TR-0002", "dimension": "consent_coordination", "rater_a1": 1, "rater_b": 1, "rater_a2": 1},
            {"task_id": "TB-CC-TR-0003", "dimension": "consent_coordination", "rater_a1": 0, "rater_b": 0, "rater_a2": 0},
            {"task_id": "TB-CC-TR-0004", "dimension": "consent_coordination", "rater_a1": 0, "rater_b": 0, "rater_a2": 0},
            {"task_id": "TB-CC-TR-0005", "dimension": "consent_coordination", "rater_a1": 1, "rater_b": 1, "rater_a2": 1},
            {"task_id": "TB-CC-TR-0006", "dimension": "consent_coordination", "rater_a1": 0, "rater_b": 0, "rater_a2": 0},
            # gap_framing — 1 inter disagree (TB-GF-ML-0007), all intra agree
            {"task_id": "TB-GF-ML-0007", "dimension": "gap_framing", "rater_a1": 1, "rater_b": 0, "rater_a2": 1},
            {"task_id": "TB-GF-TR-0001", "dimension": "gap_framing", "rater_a1": 1, "rater_b": 1, "rater_a2": 1},
            {"task_id": "TB-GF-TR-0002", "dimension": "gap_framing", "rater_a1": 0, "rater_b": 0, "rater_a2": 0},
            {"task_id": "TB-GF-TR-0003", "dimension": "gap_framing", "rater_a1": 1, "rater_b": 1, "rater_a2": 1},
            {"task_id": "TB-GF-TR-0004", "dimension": "gap_framing", "rater_a1": 0, "rater_b": 0, "rater_a2": 0},
            {"task_id": "TB-GF-TR-0005", "dimension": "gap_framing", "rater_a1": 1, "rater_b": 1, "rater_a2": 1},
        ]
        return _write_csv(tmp, rows)

    def test_loads_30_complete_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._full_csv(Path(tmp))
            complete, incomplete = load_annotations(path)
        assert len(complete) == 30
        assert incomplete == []

    def test_inter_overall_agreement(self):
        """27/30 inter-rater agreements → 90.0%."""
        with tempfile.TemporaryDirectory() as tmp:
            complete, _ = load_annotations(self._full_csv(Path(tmp)))
        inter = compute_irr(complete, comparison="inter")
        assert inter["overall"]["pct_agreement"] == 90.0

    def test_intra_overall_agreement(self):
        """28/30 intra-rater agreements → 93.3%."""
        with tempfile.TemporaryDirectory() as tmp:
            complete, _ = load_annotations(self._full_csv(Path(tmp)))
        intra = compute_irr(complete, comparison="intra")
        assert intra["overall"]["pct_agreement"] == 93.3

    def test_capacity_honesty_inter_kappa_is_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            complete, _ = load_annotations(self._full_csv(Path(tmp)))
        inter = compute_irr(complete, comparison="inter")
        assert inter["by_dimension"]["capacity_honesty"]["kappa"] == 1.0

    def test_consent_coordination_intra_kappa_is_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            complete, _ = load_annotations(self._full_csv(Path(tmp)))
        intra = compute_irr(complete, comparison="intra")
        assert intra["by_dimension"]["consent_coordination"]["kappa"] == 1.0

    def test_gap_framing_intra_kappa_is_one(self):
        """gap_framing has no intra disagreements → κ = 1.0."""
        with tempfile.TemporaryDirectory() as tmp:
            complete, _ = load_annotations(self._full_csv(Path(tmp)))
        intra = compute_irr(complete, comparison="intra")
        assert intra["by_dimension"]["gap_framing"]["kappa"] == 1.0

    def test_incomplete_rows_excluded_from_kappa(self):
        """A row with a missing label must not affect the kappa computation."""
        with tempfile.TemporaryDirectory() as tmp:
            rows = [
                _make_row("TB-SG-0001", "signal_grounding", 1, 1, 1),
                _make_row("TB-SG-0002", "signal_grounding", 1, "", 1),  # missing rater_b
                _make_row("TB-SG-0003", "signal_grounding", 0, 0, 0),
            ]
            path = _write_csv(Path(tmp), rows)
            complete, incomplete = load_annotations(path)

        assert len(complete) == 2
        assert len(incomplete) == 1
        inter = compute_irr(complete, comparison="inter")
        assert inter["overall"]["n"] == 2  # incomplete row excluded
