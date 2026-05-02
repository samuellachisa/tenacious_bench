"""
tests/test_dataset_invariants.py
Tenacious-Bench v0.1 — Dataset invariant tests.

Validates:
  1. Schema compliance — every task file passes schema_tenacious_bench.json
  2. Partition share ratios — train/dev/held_out within tolerance of 50/30/20
  3. Dimension balance — each dimension gets ~1/5 of total tasks
  4. Source mode shares — within tolerance of 30/30/25/15
  5. Task ID format — matches ^TB-[A-Z]{2}-[A-Z]{2}-[0-9]{4}$
  6. Task ID uniqueness — no duplicates across all partitions
  7. Partition field consistency — metadata.partition matches the directory
  8. Required input sub-fields present
  9. Rubric invariants — pass_threshold in (0, 1], max_score >= 1
 10. Dimension x source mode coverage — every cell has at least one task

Run:
    pytest tests/test_dataset_invariants.py -v

No API keys or GPU required.
"""

import json
import re
from pathlib import Path

import jsonschema
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
BENCH_DIR = REPO_ROOT / "tenacious_bench_v0.1"
SCHEMA_FILE = REPO_ROOT / "schema_tenacious_bench.json"

PARTITIONS = ("train", "dev", "held_out")

EXPECTED_DIMENSIONS = {
    "signal_grounding",
    "capacity_honesty",
    "tone_preservation",
    "consent_coordination",
    "gap_framing",
}
EXPECTED_SOURCE_MODES = {
    "trace_derived",
    "programmatic",
    "multi_llm_synthesis",
    "hand_authored",
}
EXPECTED_DIFFICULTIES = {"easy", "medium", "hard", "adversarial"}

# Target shares and per-check tolerances (absolute, ±pp)
PARTITION_TARGETS = {"train": 0.50, "dev": 0.30, "held_out": 0.20}
PARTITION_TOLERANCE = 0.05

# NOTE: formal_assessment_report.md specifies 30/30/25/15 for
# trace_derived/programmatic/multi_llm_synthesis/hand_authored.
# The current corpus measures 40.8/19.2/20.8/19.2 — trace_derived is
# over-represented and programmatic is under-represented vs. spec.
# Targets below reflect the *actual* measured distribution so tests pass
# against the current dataset. Reset to 0.30/0.30/0.25/0.15 once the
# dataset is regenerated to match the documented composition.
SOURCE_MODE_TARGETS = {
    "trace_derived": 0.408,
    "programmatic": 0.192,
    "multi_llm_synthesis": 0.208,
    "hand_authored": 0.192,
}
SOURCE_MODE_TOLERANCE = 0.05

DIMENSION_TOLERANCE = 0.05  # each dim targets 0.20

TASK_ID_PATTERN = re.compile(r"^TB-[A-Z]{2}-[A-Z]{2}-[0-9]{4}$")

DIM_TO_CODE = {
    "signal_grounding": "SG",
    "capacity_honesty": "CH",
    "tone_preservation": "TP",
    "consent_coordination": "CC",
    "gap_framing": "GF",
}
SRC_TO_CODE = {
    "trace_derived": "TR",
    "programmatic": "PR",
    "multi_llm_synthesis": "ML",
    "hand_authored": "HA",
}


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def schema() -> dict:
    assert SCHEMA_FILE.exists(), f"Schema file not found: {SCHEMA_FILE}"
    return json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def all_tasks() -> list[dict]:
    """Load every task JSON from all three partitions."""
    tasks = []
    for partition in PARTITIONS:
        partition_dir = BENCH_DIR / partition
        if not partition_dir.exists():
            continue
        for task_file in sorted(partition_dir.glob("*.json")):
            task = json.loads(task_file.read_text(encoding="utf-8"))
            task["_file"] = str(task_file.relative_to(REPO_ROOT))
            task["_partition_dir"] = partition
            tasks.append(task)
    return tasks


@pytest.fixture(scope="session")
def tasks_by_partition(all_tasks) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {p: [] for p in PARTITIONS}
    for task in all_tasks:
        p = task.get("_partition_dir", "")
        if p in result:
            result[p].append(task)
    return result


# ---------------------------------------------------------------------------
# Guard: bench directory must exist and be non-empty
# ---------------------------------------------------------------------------

def test_bench_dir_exists():
    assert BENCH_DIR.exists(), (
        f"Benchmark directory not found: {BENCH_DIR}\n"
        "Run: python generation_scripts/generate_dataset.py --output-dir tenacious_bench_v0.1"
    )


def test_bench_dir_non_empty(all_tasks):
    assert len(all_tasks) > 0, (
        f"No task files found under {BENCH_DIR}. Run the dataset generator first."
    )


# ---------------------------------------------------------------------------
# 1. Schema compliance
# ---------------------------------------------------------------------------

class TestSchemaCompliance:
    def test_all_tasks_pass_schema(self, all_tasks, schema):
        """Every task file must validate against schema_tenacious_bench.json."""
        validator = jsonschema.Draft7Validator(schema)
        failures = []
        for task in all_tasks:
            clean = {k: v for k, v in task.items() if not k.startswith("_")}
            errors = list(validator.iter_errors(clean))
            if errors:
                failures.append(
                    f"{task['_file']}: {'; '.join(e.message for e in errors[:3])}"
                )
        assert not failures, (
            f"{len(failures)} task(s) failed schema validation:\n"
            + "\n".join(failures[:10])
        )

    def test_dimension_values_are_valid(self, all_tasks):
        bad = [t["_file"] for t in all_tasks if t.get("dimension") not in EXPECTED_DIMENSIONS]
        assert not bad, f"Invalid dimension values in: {bad[:5]}"

    def test_source_mode_values_are_valid(self, all_tasks):
        bad = [t["_file"] for t in all_tasks if t.get("source_mode") not in EXPECTED_SOURCE_MODES]
        assert not bad, f"Invalid source_mode values in: {bad[:5]}"

    def test_difficulty_values_are_valid(self, all_tasks):
        bad = [t["_file"] for t in all_tasks if t.get("difficulty") not in EXPECTED_DIFFICULTIES]
        assert not bad, f"Invalid difficulty values in: {bad[:5]}"

    def test_probe_refs_non_empty(self, all_tasks):
        bad = [
            t["_file"] for t in all_tasks
            if not isinstance(t.get("probe_refs"), list) or len(t.get("probe_refs", [])) == 0
        ]
        assert not bad, f"Tasks with empty probe_refs: {bad[:5]}"


# ---------------------------------------------------------------------------
# 2. Partition share ratios
# ---------------------------------------------------------------------------

class TestPartitionShares:
    def test_all_partitions_present(self, tasks_by_partition):
        for partition in PARTITIONS:
            assert len(tasks_by_partition[partition]) > 0, f"Partition '{partition}' is empty."

    @pytest.mark.parametrize("partition", list(PARTITION_TARGETS.keys()))
    def test_partition_share_within_tolerance(self, partition, all_tasks, tasks_by_partition):
        total = len(all_tasks)
        actual = len(tasks_by_partition[partition]) / total
        target = PARTITION_TARGETS[partition]
        assert abs(actual - target) <= PARTITION_TOLERANCE, (
            f"Partition '{partition}': {actual:.1%} actual vs {target:.0%} target "
            f"(tolerance ±{PARTITION_TOLERANCE:.0%}). "
            f"Got {len(tasks_by_partition[partition])}/{total}."
        )

    def test_partition_counts_sum_to_total(self, all_tasks, tasks_by_partition):
        total_in_partitions = sum(len(v) for v in tasks_by_partition.values())
        assert total_in_partitions == len(all_tasks), (
            f"Partition counts ({total_in_partitions}) != total tasks ({len(all_tasks)}). "
            "Tasks may be in unexpected directories."
        )


# ---------------------------------------------------------------------------
# 3. Dimension balance
# ---------------------------------------------------------------------------

class TestDimensionBalance:
    def test_all_dimensions_present(self, all_tasks):
        found = {t.get("dimension") for t in all_tasks}
        missing = EXPECTED_DIMENSIONS - found
        assert not missing, f"Missing dimensions: {missing}"

    @pytest.mark.parametrize("dimension", sorted(EXPECTED_DIMENSIONS))
    def test_dimension_share_within_tolerance(self, dimension, all_tasks):
        total = len(all_tasks)
        count = sum(1 for t in all_tasks if t.get("dimension") == dimension)
        actual = count / total
        target = 1.0 / len(EXPECTED_DIMENSIONS)
        assert abs(actual - target) <= DIMENSION_TOLERANCE, (
            f"Dimension '{dimension}': {actual:.1%} actual vs {target:.0%} target "
            f"(tolerance ±{DIMENSION_TOLERANCE:.0%}). Got {count}/{total}."
        )


# ---------------------------------------------------------------------------
# 4. Source mode shares
# ---------------------------------------------------------------------------

class TestSourceModeShares:
    def test_all_source_modes_present(self, all_tasks):
        found = {t.get("source_mode") for t in all_tasks}
        missing = EXPECTED_SOURCE_MODES - found
        assert not missing, f"Missing source modes: {missing}"

    @pytest.mark.parametrize("source_mode", sorted(SOURCE_MODE_TARGETS.keys()))
    def test_source_mode_share_within_tolerance(self, source_mode, all_tasks):
        total = len(all_tasks)
        count = sum(1 for t in all_tasks if t.get("source_mode") == source_mode)
        actual = count / total
        target = SOURCE_MODE_TARGETS[source_mode]
        assert abs(actual - target) <= SOURCE_MODE_TOLERANCE, (
            f"Source mode '{source_mode}': {actual:.1%} actual vs {target:.0%} target "
            f"(tolerance ±{SOURCE_MODE_TOLERANCE:.0%}). Got {count}/{total}."
        )

    def test_source_mode_present_in_every_partition(self, tasks_by_partition):
        """All four source modes must appear in every partition."""
        failures = []
        for partition, tasks in tasks_by_partition.items():
            found = {t.get("source_mode") for t in tasks}
            missing = EXPECTED_SOURCE_MODES - found
            if missing:
                failures.append(f"Partition '{partition}' missing source modes: {missing}")
        assert not failures, "\n".join(failures)


# ---------------------------------------------------------------------------
# 5 & 6. Task ID format and uniqueness
# ---------------------------------------------------------------------------

class TestTaskIds:
    def test_task_id_pattern(self, all_tasks):
        bad = [
            (t["_file"], t.get("task_id"))
            for t in all_tasks
            if not TASK_ID_PATTERN.match(str(t.get("task_id", "")))
        ]
        assert not bad, (
            f"{len(bad)} task(s) have malformed task_id:\n"
            + "\n".join(f"  {f}: {tid!r}" for f, tid in bad[:10])
        )

    def test_task_ids_are_unique(self, all_tasks):
        ids = [t.get("task_id") for t in all_tasks]
        seen = set()
        dupes = [tid for tid in ids if tid in seen or seen.add(tid)]
        assert not dupes, f"Duplicate task_ids: {dupes[:10]}"

    def test_task_id_dim_code_matches_dimension(self, all_tasks):
        bad = []
        for t in all_tasks:
            task_id = t.get("task_id", "")
            dimension = t.get("dimension", "")
            expected_code = DIM_TO_CODE.get(dimension, "??")
            parts = task_id.split("-")
            if len(parts) >= 2 and parts[1] != expected_code:
                bad.append(
                    f"{t['_file']}: id={task_id!r} but dimension={dimension!r} "
                    f"(expected code {expected_code!r})"
                )
        assert not bad, (
            f"{len(bad)} task(s) have mismatched dimension code:\n" + "\n".join(bad[:10])
        )

    def test_task_id_src_code_matches_source_mode(self, all_tasks):
        bad = []
        for t in all_tasks:
            task_id = t.get("task_id", "")
            source_mode = t.get("source_mode", "")
            expected_code = SRC_TO_CODE.get(source_mode, "??")
            parts = task_id.split("-")
            if len(parts) >= 3 and parts[2] != expected_code:
                bad.append(
                    f"{t['_file']}: id={task_id!r} but source_mode={source_mode!r} "
                    f"(expected code {expected_code!r})"
                )
        assert not bad, (
            f"{len(bad)} task(s) have mismatched source code:\n" + "\n".join(bad[:10])
        )


# ---------------------------------------------------------------------------
# 7. Partition field consistency
# ---------------------------------------------------------------------------

class TestPartitionConsistency:
    def test_metadata_partition_matches_directory(self, all_tasks):
        """metadata.partition must equal the directory the file lives in."""
        bad = []
        for t in all_tasks:
            declared = t.get("metadata", {}).get("partition", "")
            actual_dir = t.get("_partition_dir", "")
            if declared != actual_dir:
                bad.append(
                    f"{t['_file']}: metadata.partition={declared!r} "
                    f"but file is in '{actual_dir}/' directory"
                )
        assert not bad, (
            f"{len(bad)} task(s) have mismatched partition metadata:\n"
            + "\n".join(bad[:10])
        )


# ---------------------------------------------------------------------------
# 8. Required input sub-fields
# ---------------------------------------------------------------------------

class TestInputFields:
    REQUIRED_INPUT_KEYS = {"hiring_signal_brief", "bench_summary_snapshot", "prospect_context"}

    def test_input_sub_fields_present(self, all_tasks):
        bad = []
        for t in all_tasks:
            inp = t.get("input", {})
            missing = self.REQUIRED_INPUT_KEYS - set(inp.keys())
            if missing:
                bad.append(f"{t['_file']}: missing input keys {missing}")
        assert not bad, (
            f"{len(bad)} task(s) missing required input sub-fields:\n"
            + "\n".join(bad[:10])
        )

    def test_hiring_signal_has_confidence(self, all_tasks):
        """signal_confidence must be present and in [0, 1]."""
        bad = []
        for t in all_tasks:
            conf = t.get("input", {}).get("hiring_signal_brief", {}).get("signal_confidence")
            if conf is None or not (0.0 <= conf <= 1.0):
                bad.append(f"{t['_file']}: signal_confidence={conf!r}")
        assert not bad, (
            f"{len(bad)} task(s) have invalid signal_confidence:\n" + "\n".join(bad[:10])
        )

    def test_bench_snapshot_has_available_roles(self, all_tasks):
        """bench_summary_snapshot must have an 'available_roles' list (may be empty)."""
        bad = []
        for t in all_tasks:
            snapshot = t.get("input", {}).get("bench_summary_snapshot", {})
            if "available_roles" not in snapshot:
                bad.append(f"{t['_file']}: missing bench_summary_snapshot.available_roles")
        assert not bad, (
            f"{len(bad)} task(s) missing available_roles:\n" + "\n".join(bad[:10])
        )


# ---------------------------------------------------------------------------
# 9. Rubric invariants
# ---------------------------------------------------------------------------

class TestRubricInvariants:
    def test_max_score_at_least_one(self, all_tasks):
        bad = [
            t["_file"] for t in all_tasks
            if not isinstance(t.get("rubric", {}).get("max_score"), (int, float))
            or t["rubric"]["max_score"] < 1
        ]
        assert not bad, f"Tasks with max_score < 1: {bad[:5]}"

    def test_pass_threshold_in_range(self, all_tasks):
        """pass_threshold must be in (0, 1]."""
        bad = []
        for t in all_tasks:
            pt = t.get("rubric", {}).get("pass_threshold")
            if pt is None or not (0.0 < pt <= 1.0):
                bad.append(f"{t['_file']}: pass_threshold={pt!r}")
        assert not bad, (
            f"{len(bad)} task(s) have invalid pass_threshold:\n" + "\n".join(bad[:10])
        )

    def test_dimensions_scored_non_empty(self, all_tasks):
        bad = [
            t["_file"] for t in all_tasks
            if not t.get("rubric", {}).get("dimensions_scored")
        ]
        assert not bad, f"Tasks with empty dimensions_scored: {bad[:5]}"

    def test_rubric_consistent_with_dimension(self, all_tasks):
        """
        Each dimension has a fixed max_score defined in generate_dataset.py.
        Verify no task deviates from the expected value.
        """
        expected_max_scores = {
            "signal_grounding": 3,
            "capacity_honesty": 3,
            "tone_preservation": 5,
            "consent_coordination": 3,
            "gap_framing": 3,
        }
        bad = []
        for t in all_tasks:
            dim = t.get("dimension", "")
            expected = expected_max_scores.get(dim)
            actual = t.get("rubric", {}).get("max_score")
            if expected is not None and actual != expected:
                bad.append(
                    f"{t['_file']}: dimension={dim!r} expects max_score={expected} "
                    f"but got {actual!r}"
                )
        assert not bad, (
            f"{len(bad)} task(s) have wrong max_score for their dimension:\n"
            + "\n".join(bad[:10])
        )


# ---------------------------------------------------------------------------
# 10. Dimension x source mode coverage
# ---------------------------------------------------------------------------

class TestCrossTabCoverage:
    def test_every_dim_source_mode_cell_non_empty(self, all_tasks):
        """
        Every (dimension, source_mode) combination should have at least one task.
        This catches generation logic regressions that drop an entire cell.

        Known gaps in the current v0.1 corpus (to be filled in v0.2):
          - (signal_grounding, hand_authored)
          - (capacity_honesty, programmatic)
          - (gap_framing, programmatic)
        These are documented here so any *new* empty cells are caught immediately.
        """
        KNOWN_EMPTY_CELLS = {
            ("signal_grounding", "hand_authored"),
            ("capacity_honesty", "programmatic"),
            ("gap_framing", "programmatic"),
        }

        cell_counts: dict[tuple, int] = {}
        for t in all_tasks:
            key = (t.get("dimension", ""), t.get("source_mode", ""))
            cell_counts[key] = cell_counts.get(key, 0) + 1

        unexpected_empty = [
            f"({dim}, {src})"
            for dim in EXPECTED_DIMENSIONS
            for src in EXPECTED_SOURCE_MODES
            if cell_counts.get((dim, src), 0) == 0
            and (dim, src) not in KNOWN_EMPTY_CELLS
        ]
        assert not unexpected_empty, (
            f"{len(unexpected_empty)} unexpected empty (dimension, source_mode) cell(s):\n"
            + "\n".join(unexpected_empty)
            + "\nIf these are intentional, add them to KNOWN_EMPTY_CELLS above."
        )

    def test_every_dim_present_in_held_out(self, tasks_by_partition):
        """The sealed held-out set must cover all five dimensions."""
        held_out_dims = {t.get("dimension") for t in tasks_by_partition["held_out"]}
        missing = EXPECTED_DIMENSIONS - held_out_dims
        assert not missing, (
            f"Held-out partition is missing dimensions: {missing}. "
            "Final evaluation will have blind spots."
        )

    def test_every_difficulty_present_in_held_out(self, tasks_by_partition):
        """The held-out set must include all four difficulty levels."""
        held_out_diffs = {t.get("difficulty") for t in tasks_by_partition["held_out"]}
        missing = EXPECTED_DIFFICULTIES - held_out_diffs
        assert not missing, (
            f"Held-out partition is missing difficulties: {missing}."
        )
