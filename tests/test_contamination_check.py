"""
tests/test_contamination_check.py
Synthetic unit tests for contamination_check.py.

Each test intentionally triggers exactly one violation type so that a silent
failure in the reporting logic is immediately visible.  No external API calls
or file I/O are required — all fixtures are built in-memory.

Violation types covered:
  run_check()
    1. ngram_exact_match          — n-gram overlap >= 1.0 (exact copy)
    2. high_tfidf_cosine          — TF-IDF cosine >= 0.85 (near-duplicate)
    3. clean task                 — no false positive on unrelated text

  run_time_shift_check()
    4. missing_created_at         — metadata.created_at absent
    5. created_before_cutoff      — created_at < cutoff_date
    6. invalid_created_at         — created_at not parseable as ISO date
    7. held_out_not_jittered      — held-out task missing bench_snapshot_jittered
    8. stale_capacity_locked_until — capacity_locked_until < created_at
    9. clean time-shift           — all rules satisfied, no violations

  Low-level utilities
   10. ngram_overlap              — exact match returns 1.0, disjoint returns 0.0
   11. build_idf + _cosine        — identical docs score 1.0, disjoint score 0.0
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from contamination_check import (
    build_idf,
    ngram_overlap,
    run_check,
    run_time_shift_check,
    task_to_text,
    _cosine,
    _tfidf_vector,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SENTINEL = object()  # distinguishes "not passed" from an explicit empty dict


def _make_task(
    task_id: str = "TB-TEST-0001",
    dimension: str = "signal_grounding",
    ground_truth: dict | None = None,
    input_data: dict | None = None,
    metadata=_SENTINEL,
) -> dict:
    """Return a minimal valid task dict.

    Pass ``metadata={}`` to produce a task with an empty metadata block
    (used to test missing_created_at).  Omit ``metadata`` to get the
    default valid metadata.
    """
    if metadata is _SENTINEL:
        metadata = {"created_at": "2026-04-29", "partition": "train"}
    return {
        "task_id": task_id,
        "dimension": dimension,
        "ground_truth": ground_truth or {"expected": ["confirm capacity"]},
        "input": input_data or {
            "bench_summary_snapshot": {
                "capacity_locked_until": "2026-12-31",
            }
        },
        "metadata": metadata,
    }


def _write_tasks(tmp_dir: Path, split: str, tasks: list[dict]) -> None:
    """Write task dicts as JSON files under tmp_dir/<split>/."""
    split_dir = tmp_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        path = split_dir / f"{task['task_id']}.json"
        path.write_text(json.dumps(task), encoding="utf-8")


# ---------------------------------------------------------------------------
# Low-level utility tests (10, 11)
# ---------------------------------------------------------------------------

class TestNgramOverlap:
    """Test 10 — ngram_overlap utility."""

    def test_exact_match_returns_one(self):
        text = "the quick brown fox jumps over the lazy dog"
        assert ngram_overlap(text, text, n=8) == 1.0

    def test_disjoint_returns_zero(self):
        a = "alpha beta gamma delta epsilon zeta eta theta"
        b = "one two three four five six seven eight nine"
        assert ngram_overlap(a, b, n=8) == 0.0

    def test_partial_overlap(self):
        a = "a b c d e f g h i j"
        b = "a b c d e f g h x y"  # first 8-gram matches, last two differ
        score = ngram_overlap(a, b, n=8)
        assert 0.0 < score < 1.0

    def test_too_short_returns_zero(self):
        # Fewer tokens than n → no n-grams → 0.0
        assert ngram_overlap("short text", "short text", n=8) == 0.0


class TestTfidfCosine:
    """Test 11 — build_idf + _cosine utilities."""

    def test_identical_docs_score_one(self):
        doc = "capacity bench signal grounding ml engineer"
        idf = build_idf([doc, doc])
        v = _tfidf_vector(doc, idf)
        assert abs(_cosine(v, v) - 1.0) < 1e-9

    def test_disjoint_docs_score_zero(self):
        doc_a = "alpha beta gamma delta"
        doc_b = "one two three four"
        idf = build_idf([doc_a, doc_b])
        va = _tfidf_vector(doc_a, idf)
        vb = _tfidf_vector(doc_b, idf)
        assert _cosine(va, vb) == 0.0

    def test_empty_vector_returns_zero(self):
        idf = build_idf(["hello world"])
        empty = {}
        v = _tfidf_vector("hello world", idf)
        assert _cosine(empty, v) == 0.0
        assert _cosine(v, empty) == 0.0


# ---------------------------------------------------------------------------
# run_check() tests (1, 2, 3)
# ---------------------------------------------------------------------------

class TestRunCheck:
    """Tests for the n-gram + TF-IDF contamination check."""

    def _run(self, tasks: list[dict], reference_texts: list[str], **kwargs) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            bench_dir = Path(tmp)
            _write_tasks(bench_dir, "train", tasks)
            return run_check(bench_dir, reference_texts, **kwargs)

    # ── Test 1: ngram_exact_match ─────────────────────────────────────────

    def test_exact_copy_triggers_ngram_flag(self):
        """A task whose text is an exact copy of a reference doc must be flagged
        with ngram_exact_match=True."""
        task = _make_task(task_id="TB-EXACT-0001")
        task_text = task_to_text(task)

        result = self._run([task], reference_texts=[task_text], ngram_n=8)

        assert result["status"] == "CONTAMINATED", (
            "Expected CONTAMINATED but got CLEAN — ngram_exact_match not triggered"
        )
        assert result["violations"], "violations list must not be empty"
        v = result["violations"][0]
        assert v["flags"]["ngram_exact_match"], (
            "ngram_exact_match flag must be True for an exact copy"
        )
        assert v["task_id"] == "TB-EXACT-0001"

    # ── Test 2: high_tfidf_cosine ─────────────────────────────────────────

    def test_near_duplicate_triggers_cosine_flag(self):
        """A task that is a near-duplicate of a reference doc (same tokens,
        slightly shuffled) must be flagged with high_tfidf_cosine=True."""
        # Build a long, distinctive reference text so TF-IDF cosine is high
        shared_tokens = " ".join([f"uniquetoken{i}" for i in range(60)])
        task = _make_task(
            task_id="TB-COSINE-0001",
            ground_truth={"expected": [shared_tokens]},
        )
        # Reference is the same token soup — cosine will be ~1.0
        reference = shared_tokens

        result = self._run([task], reference_texts=[reference], cosine_threshold=0.85)

        assert result["status"] == "CONTAMINATED", (
            "Expected CONTAMINATED — high_tfidf_cosine not triggered"
        )
        v = result["violations"][0]
        assert v["flags"]["high_tfidf_cosine"], (
            "high_tfidf_cosine flag must be True for a near-duplicate"
        )
        assert v["max_cosine_similarity_tfidf"] >= 0.85

    # ── Test 3: clean task — no false positive ────────────────────────────

    def test_clean_task_is_not_flagged(self):
        """A task with no overlap against the reference corpus must be CLEAN."""
        task = _make_task(
            task_id="TB-CLEAN-0001",
            ground_truth={"expected": ["confirm capacity for ml engineer role"]},
        )
        # Reference is completely unrelated
        reference = " ".join([f"zz{i}" for i in range(80)])

        result = self._run([task], reference_texts=[reference])

        assert result["status"] == "CLEAN", (
            f"Expected CLEAN but got CONTAMINATED — false positive. "
            f"Violations: {result.get('violations')}"
        )
        assert result["violations"] == []


# ---------------------------------------------------------------------------
# run_time_shift_check() tests (4–9)
# ---------------------------------------------------------------------------

CUTOFF = "2026-04-21"


class TestRunTimeShiftCheck:
    """Tests for the temporal ordering / jitter audit check."""

    def _run(self, tasks_by_split: dict[str, list[dict]], cutoff: str = CUTOFF) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            bench_dir = Path(tmp)
            for split, tasks in tasks_by_split.items():
                _write_tasks(bench_dir, split, tasks)
            return run_time_shift_check(bench_dir, cutoff_date=cutoff)

    def _violation_types(self, result: dict) -> list[str]:
        return [v["violation"] for v in result.get("hard_violations", [])]

    # ── Test 4: missing_created_at ────────────────────────────────────────

    def test_missing_created_at_is_flagged(self):
        """A task with no metadata.created_at must produce a missing_created_at
        hard violation."""
        task = _make_task(task_id="TB-TS-MISSING", metadata={})  # no created_at
        result = self._run({"train": [task]})

        assert "missing_created_at" in self._violation_types(result), (
            "missing_created_at violation not reported"
        )
        assert result["status"] == "VIOLATIONS"

    # ── Test 5: created_before_cutoff ─────────────────────────────────────

    def test_created_before_cutoff_is_flagged(self):
        """A task created before the cutoff date must produce a
        created_before_cutoff hard violation."""
        task = _make_task(
            task_id="TB-TS-EARLY",
            metadata={"created_at": "2025-01-01", "partition": "train"},
        )
        result = self._run({"train": [task]})

        assert "created_before_cutoff" in self._violation_types(result), (
            "created_before_cutoff violation not reported"
        )
        assert result["status"] == "VIOLATIONS"

    # ── Test 6: invalid_created_at ────────────────────────────────────────

    def test_invalid_created_at_is_flagged(self):
        """A task with a non-ISO created_at string must produce an
        invalid_created_at hard violation."""
        task = _make_task(
            task_id="TB-TS-BADDATE",
            metadata={"created_at": "not-a-date", "partition": "train"},
        )
        result = self._run({"train": [task]})

        assert "invalid_created_at" in self._violation_types(result), (
            "invalid_created_at violation not reported"
        )

    # ── Test 7: held_out_not_jittered ─────────────────────────────────────

    def test_held_out_without_jitter_produces_warning(self):
        """A held-out task missing bench_snapshot_jittered=True must produce a
        held_out_not_jittered warning (not a hard violation)."""
        task = _make_task(
            task_id="TB-TS-NOJITTER",
            metadata={
                "created_at": "2026-04-29",
                "partition": "held_out",
                # bench_snapshot_jittered intentionally absent
            },
        )
        result = self._run({"held_out": [task]})

        warning_types = [w["violation"] for w in result.get("warnings", [])]
        assert "held_out_not_jittered" in warning_types, (
            "held_out_not_jittered warning not reported"
        )
        # Must be a warning, not a hard violation
        assert "held_out_not_jittered" not in self._violation_types(result), (
            "held_out_not_jittered must be a warning, not a hard violation"
        )
        # Jitter flag should be False
        assert result["held_out_jitter_ok"] is False

    # ── Test 8: stale capacity_locked_until ───────────────────────────────

    def test_stale_capacity_locked_until_is_reported(self):
        """A task whose capacity_locked_until predates its created_at must
        appear in stale_snapshots."""
        task = _make_task(
            task_id="TB-TS-STALE",
            input_data={
                "bench_summary_snapshot": {
                    "capacity_locked_until": "2026-01-01",  # before created_at
                }
            },
            metadata={"created_at": "2026-04-29", "partition": "train"},
        )
        result = self._run({"train": [task]})

        stale_ids = [s["task_id"] for s in result.get("stale_snapshots", [])]
        assert "TB-TS-STALE" in stale_ids, (
            "Stale capacity_locked_until not reported in stale_snapshots"
        )
        assert result["status"] == "VIOLATIONS"

    # ── Test 9: clean time-shift — no violations ──────────────────────────

    def test_clean_task_passes_all_time_shift_rules(self):
        """A fully compliant task must produce no violations, no warnings,
        and no stale snapshots."""
        task = _make_task(
            task_id="TB-TS-CLEAN",
            input_data={
                "bench_summary_snapshot": {
                    "capacity_locked_until": "2026-12-31",  # future relative to created_at
                }
            },
            metadata={
                "created_at": "2026-04-29",
                "partition": "train",
                # Not held_out, so jitter rule doesn't apply
            },
        )
        result = self._run({"train": [task]})

        assert result["status"] == "CLEAN", (
            f"Expected CLEAN but got VIOLATIONS. "
            f"Hard violations: {result.get('hard_violations')}. "
            f"Stale: {result.get('stale_snapshots')}"
        )
        assert result["hard_violations"] == []
        assert result["stale_snapshots"] == []

    def test_clean_held_out_with_jitter_passes(self):
        """A held-out task with bench_snapshot_jittered=True must not produce
        any warnings or violations."""
        task = _make_task(
            task_id="TB-TS-JITTERED",
            input_data={
                "bench_summary_snapshot": {
                    "capacity_locked_until": "2026-12-31",
                }
            },
            metadata={
                "created_at": "2026-04-29",
                "partition": "held_out",
                "bench_snapshot_jittered": True,
            },
        )
        result = self._run({"held_out": [task]})

        assert result["hard_violations"] == [], (
            f"Unexpected hard violations: {result['hard_violations']}"
        )
        assert result["warnings"] == [], (
            f"Unexpected warnings: {result['warnings']}"
        )
        assert result["held_out_jitter_ok"] is True
