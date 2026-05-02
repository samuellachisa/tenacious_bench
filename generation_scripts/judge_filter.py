"""
generation_scripts/judge_filter.py
Tenacious-Bench v0.1 — Post-generation quality filter.

Applies a judge model to filter generated tasks before inclusion in the benchmark.
This is a standalone filter that can be applied to any task directory.

Judge model must be a different family from the generator models to prevent
preference leakage (Li et al. 2025).

## Scoring Modes

### Pointwise scoring (default)
Each task is scored independently on four criteria (0–10 scale):
  1. Realism (0–3)
  2. Difficulty calibration (0–2)
  3. Ground truth quality (0–3)
  4. Dimension alignment (0–2)

Tasks scoring >= threshold (default 7) are passed.

### Pairwise comparison (--pairwise)
Two tasks are compared head-to-head on the same four criteria.
The judge selects the better task and provides a preference score (0–3):
  0 = tie / too close to call
  1 = slight preference for task A or B
  2 = clear preference
  3 = strong preference

Pairwise mode is used to:
  a) Break ties when two tasks score identically in pointwise mode.
  b) Select the better task when deduplicating near-duplicates (cosine >= 0.90).
  c) Validate that hard-authored adversarial tasks are harder than programmatic ones.

Anti-leakage: the judge model must be a different family from the generator.
  - Generators: DeepSeek, Qwen, Llama (cheap-tier via OpenRouter)
  - Judge: Google Gemini (different family, non-OpenAI, non-DeepSeek)

## Judge Tier Separation (Li et al., 2025 anti-leakage policy)

The full pipeline enforces four-tier model-family separation:

  Tier 1 — Generation:    DeepSeek V3 / Qwen 2.5-72B / Llama 3.1-70B
                          (bulk task synthesis, cheap tier)
  Tier 2 — Quality filter: Google Gemini 2.0 Flash  ← THIS SCRIPT
                          (judge_filter.py, orthogonal to all Tier 1 families)
  Tier 3 — Spot-check:    Claude Haiku / GPT-4.1-mini
                          (10% sample cross-check, mid tier)
  Tier 4 — Held-out eval: Google Gemini 2.5 Flash Lite
                          (scoring_evaluator.py, sealed slice)

Invariant: Tier 1 family ∉ {Tier 2, Tier 3, Tier 4} families.
The generator never judges its own outputs at any tier.

Usage:
    # Pointwise filter (default)
    python generation_scripts/judge_filter.py \
        --input-dir tenacious_bench_v0.1/train \
        --output-dir tenacious_bench_v0.1/train_filtered \
        --judge-model google/gemini-2.0-flash-exp \
        --threshold 7

    # Pairwise comparison of two specific tasks
    python generation_scripts/judge_filter.py \
        --pairwise \
        --task-a tenacious_bench_v0.1/train/TB-CH-PR-0001.json \
        --task-b tenacious_bench_v0.1/train/TB-CH-PR-0002.json \
        --judge-model google/gemini-2.0-flash-exp

    # Pairwise tournament over a directory (selects best N tasks)
    python generation_scripts/judge_filter.py \
        --pairwise \
        --input-dir tenacious_bench_v0.1/train \
        --output-dir tenacious_bench_v0.1/train_filtered \
        --top-n 50 \
        --judge-model google/gemini-2.0-flash-exp

    # Dry run (show what would be filtered without moving files)
    python generation_scripts/judge_filter.py \
        --input-dir tenacious_bench_v0.1/train \
        --judge-model google/gemini-2.0-flash-exp \
        --threshold 7 \
        --dry-run
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# OpenRouter API client
# ---------------------------------------------------------------------------

DEFAULT_JUDGE_MODEL = "google/gemini-2.0-flash-exp"


def _call_openrouter(
    model: str,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0,
    retries: int = 3,
) -> str | None:
    """
    Call OpenRouter API with retry logic.
    Returns the response text, or None on failure.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        print("ERROR: OPENROUTER_API_KEY not set or invalid", file=sys.stderr)
        return None

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/samuellachisa/tenacious-agent",
    }

    for attempt in range(retries):
        try:
            import requests
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=90,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            is_retryable = any(
                code in str(e) for code in ("502", "503", "504", "timeout", "Timeout", "429")
            )
            if is_retryable and attempt < retries - 1:
                wait = 2 ** attempt
                print(
                    f"  [RETRY {attempt+1}/{retries}] {type(e).__name__} — retrying in {wait}s",
                    file=sys.stderr,
                )
                time.sleep(wait)
                continue
            print(f"  [ERROR] {type(e).__name__}: {e}", file=sys.stderr)
            return None
    return None


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

JUDGE_PROMPT_TEMPLATE = """You are a quality judge for Tenacious-Bench, a B2B sales agent evaluation benchmark.

Evaluate the following task for quality and dimension alignment.

## Task JSON
{task_json}

## Quality Criteria

1. **Realism (0–3):**
   - 3 = realistic company, role, signal source, bench snapshot; could be a real B2B scenario
   - 2 = mostly realistic, minor implausibilities (e.g., unusual company name, generic role)
   - 1 = generic or implausible (e.g., "Acme Corp", "Software Engineer" with no specifics)
   - 0 = nonsensical or contradictory

2. **Difficulty calibration (0–2):**
   - 2 = difficulty matches the declared level (easy/medium/hard/adversarial)
   - 1 = close but slightly off (e.g., declared "hard" but feels "medium")
   - 0 = wrong difficulty (e.g., declared "easy" but has 5-turn adversarial pushback)

3. **Ground truth quality (0–3):**
   - 3 = expected/forbidden behaviors are specific, measurable, dimension-aligned
   - 2 = mostly good, minor vagueness (e.g., "agent should be professional")
   - 1 = vague or generic (e.g., "agent should respond appropriately")
   - 0 = missing, contradictory, or wrong dimension

4. **Dimension alignment (0–2):**
   - 2 = task clearly tests the declared dimension (signal_grounding, capacity_honesty, etc.)
   - 1 = partially aligned (e.g., tests multiple dimensions, unclear focus)
   - 0 = wrong dimension (e.g., declared "capacity_honesty" but tests "tone_preservation")

**Total score: 0–10**

## Output Format

Reply with a single JSON object (no markdown, no explanation):

{{
  "realism": int,
  "difficulty_calibration": int,
  "ground_truth_quality": int,
  "dimension_alignment": int,
  "total": int,
  "pass": bool,
  "notes": "string (1–2 sentences explaining the score)"
}}

**Pass threshold:** total >= {threshold}

Evaluate now:"""


def _judge_task(task: dict, judge_model: str, threshold: int) -> dict | None:
    """Judge a task using the specified model."""
    task_json = json.dumps(task, indent=2)
    prompt = JUDGE_PROMPT_TEMPLATE.format(task_json=task_json, threshold=threshold)

    response = _call_openrouter(judge_model, prompt, max_tokens=300, temperature=0)
    if not response:
        return None

    # Extract JSON
    response = response.strip()
    if response.startswith("```json"):
        response = response[7:]
    if response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]
    response = response.strip()

    try:
        judgment = json.loads(response)
        # Ensure pass field is set
        if "pass" not in judgment:
            judgment["pass"] = judgment.get("total", 0) >= threshold
        return judgment
    except json.JSONDecodeError as e:
        print(f"  [JUDGE JSON ERROR] {e}", file=sys.stderr)
        print(f"  [RESPONSE] {response[:200]}...", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Filter pipeline
# ---------------------------------------------------------------------------

def filter_tasks(
    input_dir: Path,
    output_dir: Path | None,
    judge_model: str,
    threshold: int,
    dry_run: bool,
) -> dict:
    """
    Filter all tasks in input_dir using the judge model.
    If output_dir is provided, copy passing tasks there.
    Returns summary stats.
    """
    if not input_dir.exists():
        print(f"ERROR: input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    task_files = sorted(input_dir.glob("*.json"))
    if not task_files:
        print(f"WARNING: no JSON files found in {input_dir}", file=sys.stderr)
        return {"total": 0, "passed": 0, "failed": 0, "errors": 0}

    if output_dir and not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    passed = 0
    failed = 0
    errors = 0
    results = []

    for i, task_file in enumerate(task_files, 1):
        print(f"[{i}/{len(task_files)}] {task_file.name}...", end=" ")

        try:
            task = json.loads(task_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"ERROR (invalid JSON): {e}")
            errors += 1
            continue

        judgment = _judge_task(task, judge_model, threshold)
        if not judgment:
            print("ERROR (judge unavailable)")
            errors += 1
            continue

        task_passed = judgment.get("pass", False)
        score = judgment.get("total", 0)

        if task_passed:
            print(f"PASS (score={score}/10)")
            passed += 1
            if output_dir and not dry_run:
                # Copy task to output dir, add judge metadata
                task["metadata"] = task.get("metadata", {})
                task["metadata"]["judge_model"] = judge_model
                task["metadata"]["judge_score"] = score
                task["metadata"]["judge_notes"] = judgment.get("notes", "")
                output_file = output_dir / task_file.name
                output_file.write_text(
                    json.dumps(task, indent=2, ensure_ascii=False), encoding="utf-8"
                )
        else:
            print(f"FAIL (score={score}/10) — {judgment.get('notes', '')}")
            failed += 1

        results.append({
            "task_id": task.get("task_id", task_file.stem),
            "file": task_file.name,
            "pass": task_passed,
            "score": score,
            "notes": judgment.get("notes", ""),
        })

    return {
        "total": len(task_files),
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "pass_rate": round(passed / len(task_files), 4) if task_files else 0,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Pairwise comparison
# ---------------------------------------------------------------------------

PAIRWISE_PROMPT_TEMPLATE = """You are a quality judge for Tenacious-Bench, a B2B sales agent evaluation benchmark.

Compare two tasks for the **{dimension}** dimension and decide which is higher quality.

## Task A
{task_a_json}

## Task B
{task_b_json}

## Comparison Criteria

Evaluate both tasks on the same four dimensions:
1. **Realism** — realistic company, role, signal source, bench snapshot
2. **Difficulty calibration** — difficulty matches the declared level
3. **Ground truth quality** — expected/forbidden behaviors are specific and measurable
4. **Dimension alignment** — task clearly tests the declared dimension

## Output Format

Reply with a single JSON object (no markdown, no explanation):

{{
  "winner": "A" | "B" | "tie",
  "preference_score": int,
  "rationale": "string (1–2 sentences)",
  "scores": {{
    "task_a": {{"realism": int, "difficulty_calibration": int, "ground_truth_quality": int, "dimension_alignment": int, "total": int}},
    "task_b": {{"realism": int, "difficulty_calibration": int, "ground_truth_quality": int, "dimension_alignment": int, "total": int}}
  }}
}}

**preference_score:**
  0 = tie / too close to call
  1 = slight preference for winner
  2 = clear preference for winner
  3 = strong preference for winner

Evaluate now:"""


def _pairwise_compare(
    task_a: dict,
    task_b: dict,
    judge_model: str,
) -> dict | None:
    """
    Compare two tasks head-to-head using the judge model.

    Returns a dict with winner ("A", "B", or "tie"), preference_score (0–3),
    rationale, and individual scores for each task.
    Returns None on API failure.
    """
    dimension = task_a.get("dimension", task_b.get("dimension", "unknown"))
    task_a_json = json.dumps(task_a, indent=2)
    task_b_json = json.dumps(task_b, indent=2)

    prompt = PAIRWISE_PROMPT_TEMPLATE.format(
        dimension=dimension,
        task_a_json=task_a_json,
        task_b_json=task_b_json,
    )

    response = _call_openrouter(judge_model, prompt, max_tokens=400, temperature=0)
    if not response:
        return None

    response = response.strip()
    if response.startswith("```json"):
        response = response[7:]
    if response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]
    response = response.strip()

    try:
        result = json.loads(response)
        # Normalise winner field
        winner = result.get("winner", "tie").upper()
        if winner not in ("A", "B", "TIE"):
            winner = "tie"
        result["winner"] = winner
        return result
    except json.JSONDecodeError as e:
        print(f"  [PAIRWISE JSON ERROR] {e}", file=sys.stderr)
        return None


def pairwise_tournament(
    input_dir: Path,
    output_dir: Path | None,
    judge_model: str,
    top_n: int,
    dry_run: bool,
) -> dict:
    """
    Run a pairwise tournament over all tasks in input_dir.

    Uses a simple Elo-style scoring: each task starts with score 0.
    For each pair comparison:
      - Winner gets +preference_score points
      - Loser gets -preference_score points
      - Tie: both get 0

    After all comparisons, the top_n tasks by Elo score are kept.

    For large directories (> 20 tasks), uses a round-robin within each
    dimension-difficulty cell rather than all-pairs to limit API calls.

    Returns summary stats.
    """
    if not input_dir.exists():
        print(f"ERROR: input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    task_files = sorted(input_dir.glob("*.json"))
    if not task_files:
        print(f"WARNING: no JSON files found in {input_dir}", file=sys.stderr)
        return {"total": 0, "kept": 0, "comparisons": 0}

    # Load all tasks
    tasks = []
    for tf in task_files:
        try:
            task = json.loads(tf.read_text(encoding="utf-8"))
            tasks.append({"file": tf, "task": task, "elo": 0, "wins": 0, "losses": 0, "ties": 0})
        except json.JSONDecodeError:
            print(f"  SKIP (invalid JSON): {tf.name}", file=sys.stderr)

    if not tasks:
        return {"total": 0, "kept": 0, "comparisons": 0}

    # Group by dimension-difficulty for round-robin within cells
    cells: dict[str, list[int]] = {}
    for i, t in enumerate(tasks):
        dim = t["task"].get("dimension", "unknown")
        diff = t["task"].get("difficulty", "unknown")
        key = f"{dim}/{diff}"
        cells.setdefault(key, []).append(i)

    comparisons = 0
    for cell_key, indices in cells.items():
        if len(indices) < 2:
            continue
        print(f"  [{cell_key}] {len(indices)} tasks — running {len(indices)-1} comparisons")
        # Round-robin: compare each task against the next (not all-pairs for large cells)
        for k in range(len(indices) - 1):
            i, j = indices[k], indices[k + 1]
            task_a = tasks[i]["task"]
            task_b = tasks[j]["task"]

            result = _pairwise_compare(task_a, task_b, judge_model)
            comparisons += 1

            if not result:
                print(f"    [{k+1}] ERROR — skipping pair", file=sys.stderr)
                continue

            winner = result.get("winner", "TIE")
            pref = result.get("preference_score", 0)
            print(f"    [{k+1}] {task_a.get('task_id','?')} vs {task_b.get('task_id','?')} "
                  f"→ winner={winner} pref={pref} | {result.get('rationale','')[:60]}")

            if winner == "A":
                tasks[i]["elo"] += pref
                tasks[i]["wins"] += 1
                tasks[j]["elo"] -= pref
                tasks[j]["losses"] += 1
            elif winner == "B":
                tasks[j]["elo"] += pref
                tasks[j]["wins"] += 1
                tasks[i]["elo"] -= pref
                tasks[i]["losses"] += 1
            else:
                tasks[i]["ties"] += 1
                tasks[j]["ties"] += 1

    # Sort by Elo score descending, keep top_n
    tasks.sort(key=lambda t: t["elo"], reverse=True)
    kept = tasks[:top_n]
    dropped = tasks[top_n:]

    if output_dir and not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        for entry in kept:
            task = entry["task"]
            task["metadata"] = task.get("metadata", {})
            task["metadata"]["pairwise_elo"] = entry["elo"]
            task["metadata"]["pairwise_wins"] = entry["wins"]
            task["metadata"]["pairwise_losses"] = entry["losses"]
            out_file = output_dir / entry["file"].name
            out_file.write_text(json.dumps(task, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "total": len(tasks),
        "kept": len(kept),
        "dropped": len(dropped),
        "comparisons": comparisons,
        "top_tasks": [
            {
                "task_id": t["task"].get("task_id", t["file"].stem),
                "elo": t["elo"],
                "wins": t["wins"],
                "losses": t["losses"],
            }
            for t in kept[:10]
        ],
    }


def compare_two_tasks(
    task_a_path: Path,
    task_b_path: Path,
    judge_model: str,
) -> dict | None:
    """
    Compare exactly two tasks and print the result.
    Convenience wrapper for --pairwise --task-a --task-b CLI usage.
    """
    task_a = json.loads(task_a_path.read_text(encoding="utf-8"))
    task_b = json.loads(task_b_path.read_text(encoding="utf-8"))

    print(f"Comparing:")
    print(f"  Task A: {task_a.get('task_id', task_a_path.stem)}")
    print(f"  Task B: {task_b.get('task_id', task_b_path.stem)}")
    print(f"  Judge:  {judge_model}")
    print()

    result = _pairwise_compare(task_a, task_b, judge_model)
    if not result:
        print("ERROR: pairwise comparison failed", file=sys.stderr)
        return None

    winner = result.get("winner", "TIE")
    pref = result.get("preference_score", 0)
    rationale = result.get("rationale", "")
    scores = result.get("scores", {})

    print(f"Winner: {winner}  (preference_score={pref}/3)")
    print(f"Rationale: {rationale}")
    print()
    if scores:
        print(f"{'Criterion':<30} {'Task A':>8} {'Task B':>8}")
        print(f"{'-'*30} {'-'*8} {'-'*8}")
        criteria = ["realism", "difficulty_calibration", "ground_truth_quality", "dimension_alignment", "total"]
        for c in criteria:
            a_val = scores.get("task_a", {}).get(c, "?")
            b_val = scores.get("task_b", {}).get(c, "?")
            print(f"  {c:<28} {str(a_val):>8} {str(b_val):>8}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Tenacious-Bench v0.1 — Judge filter for generated tasks"
    )

    # Mode selection
    parser.add_argument(
        "--pairwise",
        action="store_true",
        help="Run pairwise comparison mode instead of pointwise filtering",
    )

    # Pointwise args
    parser.add_argument("--input-dir", type=Path, help="Directory of tasks to filter")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to copy passing tasks (if omitted, no files are moved)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=7,
        help="Minimum score (0–10) to pass pointwise filter (default: 7)",
    )

    # Pairwise-specific args
    parser.add_argument("--task-a", type=Path, help="Path to task A (pairwise mode)")
    parser.add_argument("--task-b", type=Path, help="Path to task B (pairwise mode)")
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of top tasks to keep in pairwise tournament mode (default: 50)",
    )

    # Shared args
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help=f"OpenRouter model ID for judge (default: {DEFAULT_JUDGE_MODEL})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be filtered without moving files",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    # ── Pairwise mode ──────────────────────────────────────────────────────
    if args.pairwise:
        # Single pair comparison
        if args.task_a and args.task_b:
            result = compare_two_tasks(args.task_a, args.task_b, args.judge_model)
            if args.json and result:
                print(json.dumps(result, indent=2))
            sys.exit(0 if result else 1)

        # Tournament over directory
        if not args.input_dir:
            print("ERROR: --input-dir required for pairwise tournament mode", file=sys.stderr)
            sys.exit(1)

        if args.dry_run:
            print("DRY RUN: no files will be moved")

        print(f"Pairwise tournament mode")
        print(f"Judge model: {args.judge_model}")
        print(f"Input: {args.input_dir}")
        print(f"Top-N: {args.top_n}")
        if args.output_dir:
            print(f"Output: {args.output_dir}")
        print()

        summary = pairwise_tournament(
            args.input_dir,
            args.output_dir,
            args.judge_model,
            args.top_n,
            args.dry_run,
        )

        print()
        print(f"{'='*60}")
        print("Pairwise Tournament Summary")
        print(f"{'='*60}")
        print(f"Total tasks:   {summary['total']}")
        print(f"Kept (top-N):  {summary['kept']}")
        print(f"Dropped:       {summary['dropped']}")
        print(f"Comparisons:   {summary['comparisons']}")
        print(f"{'='*60}")
        print("Top 10 tasks by Elo score:")
        for t in summary.get("top_tasks", []):
            print(f"  {t['task_id']:<35} elo={t['elo']:+3d}  W={t['wins']} L={t['losses']}")

        if args.json:
            print()
            print(json.dumps(summary, indent=2))

        sys.exit(0)

    # ── Pointwise mode (default) ───────────────────────────────────────────
    if not args.input_dir:
        print("ERROR: --input-dir required for pointwise filter mode", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print("DRY RUN: no files will be moved")

    print(f"Pointwise filter mode")
    print(f"Judge model: {args.judge_model}")
    print(f"Pass threshold: {args.threshold}/10")
    print(f"Input: {args.input_dir}")
    if args.output_dir:
        print(f"Output: {args.output_dir}")
    print()

    summary = filter_tasks(
        args.input_dir,
        args.output_dir,
        args.judge_model,
        args.threshold,
        args.dry_run,
    )

    print()
    print(f"{'='*60}")
    print("Pointwise Filter Summary")
    print(f"{'='*60}")
    print(f"Total:      {summary['total']}")
    print(f"Passed:     {summary['passed']} ({summary['pass_rate']:.1%})")
    print(f"Failed:     {summary['failed']}")
    print(f"Errors:     {summary['errors']}")
    print(f"{'='*60}")

    if args.json:
        print()
        print(json.dumps(summary, indent=2))

    sys.exit(0 if summary["errors"] == 0 else 1)


if __name__ == "__main__":
    main()

