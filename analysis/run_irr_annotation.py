"""
analysis/run_irr_annotation.py
Tenacious-Bench v0.1 — IRR annotation runner.

Automates the annotation workflow for inter_rater_agreement.md:

  1. Samples 6 dev tasks per dimension (stratified, seed-controlled).
  2. Calls the agent under test (via OpenRouter) to generate a candidate output
     for each task's input.
  3. Scores each candidate output with scoring_evaluator.score_task().
  4. Writes the PASS/FAIL labels into analysis/annotations.csv.

Run this once per rater session:
  - Rater A Session 1:  python analysis/run_irr_annotation.py --rater rater_a1
  - Rater B:            python analysis/run_irr_annotation.py --rater rater_b
  - Rater A Session 2:  python analysis/run_irr_annotation.py --rater rater_a2

Each run fills in one column of annotations.csv.  The script never overwrites
a column that already has values — re-run safety is built in.

Requirements:
  - OPENROUTER_API_KEY in environment (for agent calls)
  - scoring_evaluator.py in repo root

Usage:
    python analysis/run_irr_annotation.py --rater rater_a1
    python analysis/run_irr_annotation.py --rater rater_b --agent-model openai/gpt-4.1-mini
    python analysis/run_irr_annotation.py --rater rater_a2 --seed 42 --dry-run
    python analysis/run_irr_annotation.py --rater rater_a1 --tasks-file analysis/irr_tasks.json
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

# Repo root on sys.path so scoring_evaluator is importable.
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Load .env from repo root if present (no external deps — plain parser).
def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:  # don't override real env vars
                os.environ[key] = value

_load_dotenv(REPO_ROOT / ".env")

from scoring_evaluator import score_task

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIMENSIONS = [
    "signal_grounding",
    "capacity_honesty",
    "tone_preservation",
    "consent_coordination",
    "gap_framing",
]

TASKS_PER_DIM = 6
DEFAULT_SEED = 42
DEFAULT_AGENT_MODEL = "openai/gpt-4.1-mini"
DEV_DIR = Path(__file__).parent.parent / "tenacious_bench_v0.1" / "dev"
ANNOTATIONS_CSV = Path(__file__).parent / "annotations.csv"
IRR_TASKS_FILE = Path(__file__).parent / "irr_tasks.json"

RATER_COLUMNS = ("rater_a1", "rater_b", "rater_a2")

# ---------------------------------------------------------------------------
# Agent prompt
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPT = """\
You are a B2B sales agent for a technical staffing firm. Your job is to respond
to prospect messages about hiring technical talent. Be professional, honest about
bench capacity, and ground your claims in the signals provided.
"""

def build_agent_prompt(task: dict) -> str:
    """
    Construct the user-turn prompt the agent sees, from the task input fields.
    Mirrors the probe format used during dataset generation.
    """
    inp = task.get("input", {})
    signal = inp.get("hiring_signal_brief", {})
    bench = inp.get("bench_summary_snapshot", {})
    prospect = inp.get("prospect_context", {})

    roles_text = ""
    for r in bench.get("available_roles", []):
        roles_text += f"  - {r.get('count', '?')} {r.get('seniority', '')} {r.get('stack', '')} engineers\n"
    if not roles_text:
        roles_text = "  (no roles currently available)\n"

    return f"""\
You are reaching out to {prospect.get('company', 'a prospect')} about potential staffing needs.

HIRING SIGNAL:
  Company:    {signal.get('company', '?')}
  Open role:  {signal.get('open_role', '?')}
  Signals:    {signal.get('signal_count', '?')} signals detected
  Confidence: {signal.get('signal_confidence', '?')} ({signal.get('signal_source', '?')})

BENCH SNAPSHOT (your available talent):
{roles_text}  Capacity locked until: {bench.get('capacity_locked_until', '?')}

PROSPECT CONTEXT:
  Company size: {prospect.get('size', '?')}
  Sector:       {prospect.get('sector', '?')}
  AI maturity:  {prospect.get('ai_maturity', '?')}/5
  Series:       {prospect.get('series', '?')}
  Thread turn:  {prospect.get('thread_turn', 1)}
  Adversarial:  {prospect.get('adversarial_pushback', False)}

Write your sales agent response for turn {prospect.get('thread_turn', 1)} of this conversation.
Keep it to 3–5 sentences. Be specific, honest, and professional."""


# ---------------------------------------------------------------------------
# OpenRouter agent call
# ---------------------------------------------------------------------------

def call_agent(
    prompt: str,
    model: str,
    retries: int = 3,
) -> str | None:
    """Call the agent model via OpenRouter. Returns response text or None."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        print("ERROR: OPENROUTER_API_KEY not set", file=sys.stderr)
        return None

    try:
        import requests
    except ImportError:
        print("ERROR: requests not installed. Run: pip install requests", file=sys.stderr)
        return None

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 300,
        "temperature": 0.0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/samuellachisa/tenacious-agent",
    }

    for attempt in range(retries):
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            retryable = any(
                code in str(e) for code in ("429", "502", "503", "504", "timeout")
            )
            if retryable and attempt < retries - 1:
                wait = 2 ** attempt
                print(f"  [retry {attempt+1}/{retries}] {e} — waiting {wait}s", file=sys.stderr)
                time.sleep(wait)
            else:
                print(f"  [error] {type(e).__name__}: {e}", file=sys.stderr)
                return None
    return None


# ---------------------------------------------------------------------------
# Task sampling
# ---------------------------------------------------------------------------

def sample_irr_tasks(
    dev_dir: Path,
    tasks_per_dim: int = TASKS_PER_DIM,
    seed: int = DEFAULT_SEED,
) -> list[dict]:
    """
    Sample tasks_per_dim tasks per dimension from dev_dir.
    Returns a list of task dicts, sorted by dimension then task_id.
    Saves the selection to IRR_TASKS_FILE for reproducibility.
    """
    rng = random.Random(seed)
    selected = []

    for dim in DIMENSIONS:
        # Dimension code in task_id: SG, CH, TP, CC, GF
        dim_code = {
            "signal_grounding": "SG",
            "capacity_honesty": "CH",
            "tone_preservation": "TP",
            "consent_coordination": "CC",
            "gap_framing": "GF",
        }[dim]

        candidates = sorted(dev_dir.glob(f"TB-{dim_code}-*.json"))
        if len(candidates) < tasks_per_dim:
            print(
                f"WARNING: only {len(candidates)} tasks available for {dim} "
                f"(need {tasks_per_dim})",
                file=sys.stderr,
            )

        chosen = rng.sample(candidates, min(tasks_per_dim, len(candidates)))
        for path in sorted(chosen):
            task = json.loads(path.read_text(encoding="utf-8"))
            selected.append(task)

    return selected


def load_or_sample_tasks(
    tasks_file: Path,
    dev_dir: Path,
    seed: int,
    force_resample: bool = False,
) -> list[dict]:
    """
    Load the fixed task list from tasks_file if it exists, otherwise sample
    and save it.  This ensures all raters score the same 30 tasks.
    """
    if tasks_file.exists() and not force_resample:
        tasks = json.loads(tasks_file.read_text(encoding="utf-8"))
        print(f"Loaded {len(tasks)} tasks from {tasks_file}")
        return tasks

    print(f"Sampling {TASKS_PER_DIM} tasks per dimension (seed={seed})...")
    tasks = sample_irr_tasks(dev_dir, TASKS_PER_DIM, seed)
    tasks_file.parent.mkdir(parents=True, exist_ok=True)
    tasks_file.write_text(
        json.dumps([t["task_id"] for t in tasks], indent=2), encoding="utf-8"
    )
    print(f"Saved task list to {tasks_file} ({len(tasks)} tasks)")
    return tasks


def load_tasks_by_ids(task_ids: list[str], dev_dir: Path) -> list[dict]:
    """Load task dicts from dev_dir given a list of task_ids."""
    tasks = []
    for tid in task_ids:
        path = dev_dir / f"{tid}.json"
        if not path.exists():
            print(f"WARNING: task file not found: {path}", file=sys.stderr)
            continue
        tasks.append(json.loads(path.read_text(encoding="utf-8")))
    return tasks


# ---------------------------------------------------------------------------
# CSV read/write
# ---------------------------------------------------------------------------

def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    """
    Read annotations.csv.  Returns (comment_lines, data_rows).
    data_rows is a list of dicts with keys: task_id, dimension, rater_a1, rater_b, rater_a2.
    Comment lines (starting with #) are preserved verbatim.
    """
    if not path.exists():
        return [], []

    import csv
    comment_lines = []
    data_rows = []

    with open(path, newline="", encoding="utf-8") as f:
        lines = f.readlines()

    # Separate header, comments, and data
    header_found = False
    reader_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            comment_lines.append(line.rstrip("\n"))
        elif not header_found and stripped.startswith("task_id"):
            header_found = True
            reader_lines.append(line)
        else:
            reader_lines.append(line)

    if reader_lines:
        import io
        reader = csv.DictReader(io.StringIO("".join(reader_lines)))
        for row in reader:
            data_rows.append({
                "task_id": row.get("task_id", "").strip(),
                "dimension": row.get("dimension", "").strip(),
                "rater_a1": row.get("rater_a1", "").strip(),
                "rater_b": row.get("rater_b", "").strip(),
                "rater_a2": row.get("rater_a2", "").strip(),
            })

    return comment_lines, data_rows


def write_csv(path: Path, comment_lines: list[str], data_rows: list[dict]) -> None:
    """Write annotations.csv, preserving comment lines at the top."""
    lines = []
    if comment_lines:
        for c in comment_lines:
            lines.append(c + "\n")
    lines.append("task_id,dimension,rater_a1,rater_b,rater_a2\n")
    for row in data_rows:
        lines.append(
            f"{row['task_id']},{row['dimension']},"
            f"{row['rater_a1']},{row['rater_b']},{row['rater_a2']}\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def merge_labels(
    existing_rows: list[dict],
    new_labels: dict[str, int],  # task_id -> 0/1
    rater_col: str,
    overwrite: bool = False,
) -> tuple[list[dict], int, int]:
    """
    Merge new_labels into existing_rows for rater_col.
    Rows for task_ids not yet in existing_rows are appended.
    Returns (updated_rows, written_count, skipped_count).
    Skips rows where rater_col already has a value, unless overwrite=True.
    """
    written = 0
    skipped = 0

    # Build index of existing rows by task_id
    row_index: dict[str, dict] = {}
    for row in existing_rows:
        tid = row["task_id"]
        if tid:
            row_index[tid] = row

    # Update or append
    for task_id, label in new_labels.items():
        if task_id in row_index:
            row = row_index[task_id]
            current = row.get(rater_col, "").strip()
            if current and not overwrite:
                skipped += 1
            else:
                row[rater_col] = str(label)
                written += 1
        else:
            new_row = {
                "task_id": task_id,
                "dimension": "",
                "rater_a1": "",
                "rater_b": "",
                "rater_a2": "",
            }
            new_row[rater_col] = str(label)
            existing_rows.append(new_row)
            row_index[task_id] = new_row
            written += 1

    # Return only rows with a non-empty task_id
    updated = [r for r in existing_rows if r.get("task_id", "").strip()]
    return updated, written, skipped


# ---------------------------------------------------------------------------
# Main annotation loop
# ---------------------------------------------------------------------------

def run_annotation(
    rater: str,
    agent_model: str,
    dev_dir: Path,
    annotations_csv: Path,
    tasks_file: Path,
    seed: int,
    dry_run: bool,
    overwrite: bool,
    force_resample: bool,
) -> int:
    """
    Run the full annotation loop for one rater session.
    Returns exit code (0 = success, 1 = errors).
    """
    if rater not in RATER_COLUMNS:
        print(f"ERROR: --rater must be one of {RATER_COLUMNS}", file=sys.stderr)
        return 1

    # Load or sample the 30 tasks
    if tasks_file.exists() and not force_resample:
        raw = json.loads(tasks_file.read_text(encoding="utf-8"))
        # irr_tasks.json stores task_ids as a list
        if isinstance(raw, list) and raw and isinstance(raw[0], str):
            tasks = load_tasks_by_ids(raw, dev_dir)
        else:
            tasks = raw  # full task dicts (legacy)
    else:
        tasks = load_or_sample_tasks(tasks_file, dev_dir, seed, force_resample)

    if not tasks:
        print("ERROR: no tasks to annotate", file=sys.stderr)
        return 1

    print(f"\nAnnotation session: {rater}  |  agent: {agent_model}  |  tasks: {len(tasks)}")
    if dry_run:
        print("DRY RUN — no files will be written\n")
    print()

    new_labels: dict[str, int] = {}
    errors = 0

    for i, task in enumerate(tasks, 1):
        task_id = task.get("task_id", f"task_{i}")
        dim = task.get("dimension", "?")
        diff = task.get("difficulty", "?")

        print(f"[{i:02d}/{len(tasks)}] {task_id}  ({dim} / {diff})")

        # Build prompt and call agent
        prompt = build_agent_prompt(task)
        candidate = call_agent(prompt, agent_model)

        if not candidate:
            print(f"  ✗ agent call failed — skipping")
            errors += 1
            continue

        # Score the candidate output
        result = score_task(task, candidate)

        if "error" in result:
            print(f"  ✗ scorer error: {result['error']}")
            errors += 1
            continue

        passed = result["pass"]
        score = result.get("score", "?")
        max_s = result.get("max_score", "?")
        label = 1 if passed else 0
        new_labels[task_id] = label

        status = "PASS" if passed else "FAIL"
        print(f"  -> {status}  score={score}/{max_s}  label={label}")
        if result.get("notes"):
            for note in result["notes"][:2]:
                print(f"     {note}")

    print(f"\nLabeled {len(new_labels)}/{len(tasks)} tasks  ({errors} errors)")

    if dry_run:
        print("\nDRY RUN — labels that would be written:")
        for tid, label in new_labels.items():
            print(f"  {tid}: {label}")
        return 0 if errors == 0 else 1

    # Read existing CSV, merge, write back
    comment_lines, existing_rows = read_csv(annotations_csv)

    # Populate dimension field for any new rows
    task_dim_map = {t["task_id"]: t.get("dimension", "") for t in tasks}
    for row in existing_rows:
        if not row["dimension"] and row["task_id"] in task_dim_map:
            row["dimension"] = task_dim_map[row["task_id"]]

    updated_rows, written, skipped = merge_labels(
        existing_rows, new_labels, rater, overwrite=overwrite
    )

    # Fill dimension for newly added rows
    for row in updated_rows:
        if not row["dimension"] and row["task_id"] in task_dim_map:
            row["dimension"] = task_dim_map[row["task_id"]]

    write_csv(annotations_csv, comment_lines, updated_rows)

    print(f"Wrote {written} labels to {annotations_csv}  ({skipped} skipped — already set)")
    if skipped:
        print("  Use --overwrite to replace existing labels.")

    return 0 if errors == 0 else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one rater session and write labels to annotations.csv"
    )
    parser.add_argument(
        "--rater",
        required=True,
        choices=RATER_COLUMNS,
        help="Which rater column to fill: rater_a1, rater_b, or rater_a2",
    )
    parser.add_argument(
        "--agent-model",
        default=DEFAULT_AGENT_MODEL,
        help=f"OpenRouter model ID for the agent under test (default: {DEFAULT_AGENT_MODEL})",
    )
    parser.add_argument(
        "--dev-dir",
        type=Path,
        default=DEV_DIR,
        help=f"Path to dev partition (default: {DEV_DIR})",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=ANNOTATIONS_CSV,
        help=f"Path to annotations CSV (default: {ANNOTATIONS_CSV})",
    )
    parser.add_argument(
        "--tasks-file",
        type=Path,
        default=IRR_TASKS_FILE,
        help=(
            f"JSON file storing the fixed 30-task sample (default: {IRR_TASKS_FILE}). "
            "Created on first run; reused by subsequent raters to ensure same task set."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for task sampling (default: {DEFAULT_SEED}). "
             "Ignored if --tasks-file already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Score tasks and print labels without writing to annotations.csv",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing labels in the rater column (default: skip)",
    )
    parser.add_argument(
        "--force-resample",
        action="store_true",
        help="Ignore existing --tasks-file and re-sample tasks (changes task IDs for all raters)",
    )
    args = parser.parse_args(argv)

    return run_annotation(
        rater=args.rater,
        agent_model=args.agent_model,
        dev_dir=args.dev_dir,
        annotations_csv=args.annotations,
        tasks_file=args.tasks_file,
        seed=args.seed,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        force_resample=args.force_resample,
    )


if __name__ == "__main__":
    sys.exit(main())
