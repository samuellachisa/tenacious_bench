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

### Calibration (--calibrate)
Samples N tasks from a previously filtered directory, re-scores them with the
eval-tier model, and computes agreement with the dev-tier scores stored in task
metadata. Reports per-criterion score drift and pass/fail flip rate.

Use this after running the dev-tier filter to validate that the cheap dev-tier
judge is not systematically biased before the held-out evaluation runs.

## Judge Tier Separation (Li et al., 2025 anti-leakage policy)

The full pipeline enforces four-tier model-family separation:

  Tier 1 — Generation:    DeepSeek V3 / Qwen 2.5-72B / Llama 3.1-70B
                          (bulk task synthesis, cheap tier)
  Tier 2 — Dev filter:    Google Gemini 2.0 Flash  <- default for --judge-model
                          (judge_filter.py pointwise/pairwise, orthogonal to Tier 1)
  Tier 3 — Calibration:   Claude Haiku / GPT-4.1-mini  <- default for --eval-model
                          (--calibrate spot-check, 10% sample, mid tier)
  Tier 4 — Held-out eval: Google Gemini 2.5 Flash Lite
                          (scoring_evaluator.py, sealed slice)

Invariant: Tier 1 family not in {Tier 2, Tier 3, Tier 4} families.
The generator never judges its own outputs at any tier.

Model constants are defined below as DEV_JUDGE_MODEL and EVAL_JUDGE_MODEL.
Pass --judge-model to override the dev-tier model.
Pass --eval-model to override the eval-tier model used in --calibrate.

Usage:
    # Pointwise filter (default, dev-tier judge)
    python generation_scripts/judge_filter.py \\
        --input-dir tenacious_bench_v0.1/train \\
        --output-dir tenacious_bench_v0.1/train_filtered \\
        --threshold 7

    # Calibration: spot-check 10% of filtered tasks with eval-tier model
    python generation_scripts/judge_filter.py \\
        --calibrate \\
        --input-dir tenacious_bench_v0.1/train_filtered \\
        --sample-n 25 \\
        --eval-model anthropic/claude-haiku-20240307

    # Pairwise comparison of two specific tasks
    python generation_scripts/judge_filter.py \\
        --pairwise \\
        --task-a tenacious_bench_v0.1/train/TB-CH-PR-0001.json \\
        --task-b tenacious_bench_v0.1/train/TB-CH-PR-0002.json

    # Pairwise tournament over a directory (selects best N tasks)
    python generation_scripts/judge_filter.py \\
        --pairwise \\
        --input-dir tenacious_bench_v0.1/train \\
        --output-dir tenacious_bench_v0.1/train_filtered \\
        --top-n 50

    # Dry run (show what would be filtered without moving files)
    python generation_scripts/judge_filter.py \\
        --input-dir tenacious_bench_v0.1/train \\
        --threshold 7 \\
        --dry-run
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Model family registry — anti-leakage enforcement
# ---------------------------------------------------------------------------
# Maps OpenRouter model IDs (or prefixes) to a canonical family name.
# Used by assert_no_family_overlap() to prevent a generator and judge from
# the same family ever being configured together (Li et al. 2025).
#
# Rules for adding entries:
#   - Use the exact OpenRouter model ID where possible.
#   - For model families with many variants, map the org-level prefix
#     (e.g. "deepseek/") so new checkpoints are covered automatically.
#   - Family names must be lowercase, hyphen-separated strings.

MODEL_FAMILIES: dict[str, str] = {
    # Tier 1 — Generation models
    "deepseek/deepseek-chat":                   "deepseek",
    "deepseek/deepseek-chat-v3-0324":           "deepseek",
    "deepseek/deepseek-r1":                     "deepseek",
    "qwen/qwen-2.5-72b-instruct":               "qwen",
    "qwen/qwen-2.5-7b-instruct":                "qwen",
    "meta-llama/llama-3.1-70b-instruct":        "meta-llama",
    "meta-llama/llama-3.1-8b-instruct":         "meta-llama",
    "meta-llama/llama-3.3-70b-instruct":        "meta-llama",
    # Tier 2 — Dev judge
    "google/gemini-2.0-flash-exp":              "google",
    "google/gemini-2.0-flash-001":              "google",
    "google/gemini-2.5-flash":                  "google",
    "google/gemini-2.5-flash-preview-05-20":    "google",
    "google/gemini-2.5-flash-lite":             "google",
    "google/gemini-2.5-pro":                    "google",
    # Tier 3 — Calibration / eval judge
    "anthropic/claude-haiku-20240307":          "anthropic",
    "anthropic/claude-3-5-haiku":               "anthropic",
    "anthropic/claude-3-5-sonnet":              "anthropic",
    "anthropic/claude-3-7-sonnet":              "anthropic",
    "openai/gpt-4.1-mini":                      "openai",
    "openai/gpt-4.1":                           "openai",
    "openai/gpt-4o":                            "openai",
    "openai/gpt-4o-mini":                       "openai",
    # Tier 4 — Held-out eval (scoring_evaluator.py)
    # Listed here so the check covers cross-script misconfigurations.
    "google/gemini-2.5-flash-lite-preview-06-17": "google",
}

# Generation models that must never appear as judge or eval models.
# Extend this list when new generator checkpoints are added to the pipeline.
GENERATION_MODELS: list[str] = [
    "deepseek/deepseek-chat",
    "deepseek/deepseek-chat-v3-0324",
    "deepseek/deepseek-r1",
    "qwen/qwen-2.5-72b-instruct",
    "qwen/qwen-2.5-7b-instruct",
    "meta-llama/llama-3.1-70b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3.3-70b-instruct",
]


def _get_family(model_id: str) -> str | None:
    """
    Return the canonical family name for a model ID, or None if unknown.

    Tries an exact match first, then falls back to a prefix match on the
    org slug (e.g. "deepseek/" covers all DeepSeek checkpoints).
    """
    if model_id in MODEL_FAMILIES:
        return MODEL_FAMILIES[model_id]
    # Prefix match: "org/model-name" → check "org/"
    org_prefix = model_id.split("/")[0] + "/"
    for key, family in MODEL_FAMILIES.items():
        if key.startswith(org_prefix):
            return family
    return None


def assert_no_family_overlap(
    judge_model: str,
    eval_model: str | None = None,
    generation_models: list[str] | None = None,
) -> None:
    """
    Hard check: raise SystemExit(1) if any judge/eval model shares a family
    with any generation model.

    This enforces the Li et al. (2025) anti-leakage invariant:
        Tier 1 family ∉ {Tier 2, Tier 3, Tier 4} families.

    Args:
        judge_model:        The Tier 2 dev-judge model ID (--judge-model).
        eval_model:         The Tier 3 eval/calibration model ID (--eval-model).
                            Pass None to skip the eval-model check.
        generation_models:  List of Tier 1 generator model IDs to check against.
                            Defaults to GENERATION_MODELS.

    Raises:
        SystemExit(1) on any family collision, printing a clear error message.
    """
    gen_models = generation_models if generation_models is not None else GENERATION_MODELS
    judge_models_to_check: list[tuple[str, str]] = [("--judge-model (Tier 2)", judge_model)]
    if eval_model:
        judge_models_to_check.append(("--eval-model (Tier 3)", eval_model))

    violations: list[str] = []

    for role_label, judge_id in judge_models_to_check:
        judge_family = _get_family(judge_id)
        if judge_family is None:
            # Unknown model — warn but don't block; the registry may be incomplete.
            print(
                f"WARNING: {role_label} model '{judge_id}' is not in MODEL_FAMILIES registry. "
                "Cannot verify family separation. Add it to MODEL_FAMILIES to enable the check.",
                file=sys.stderr,
            )
            continue

        for gen_id in gen_models:
            gen_family = _get_family(gen_id)
            if gen_family is None:
                continue
            if judge_family == gen_family:
                violations.append(
                    f"  VIOLATION: {role_label} '{judge_id}' (family={judge_family}) "
                    f"shares a family with generator '{gen_id}' (family={gen_family})."
                )

    # Also check that judge_model and eval_model are not the same family as each other
    # (Tier 2 and Tier 3 must be independent for calibration to be meaningful).
    if eval_model:
        judge_family = _get_family(judge_model)
        eval_family = _get_family(eval_model)
        if judge_family and eval_family and judge_family == eval_family:
            violations.append(
                f"  VIOLATION: --judge-model '{judge_model}' (family={judge_family}) "
                f"and --eval-model '{eval_model}' (family={eval_family}) are the same family. "
                "Calibration requires two independent model families to be meaningful."
            )

    if violations:
        print(
            "\nERROR: Model family separation violated — generator and judge from the same family.\n"
            "This breaks the anti-leakage invariant (Li et al. 2025).\n",
            file=sys.stderr,
        )
        for v in violations:
            print(v, file=sys.stderr)
        print(
            "\nFix: choose a judge/eval model from a different family than the generation models.\n"
            "Generation model families in use: "
            + ", ".join(sorted({_get_family(m) for m in gen_models if _get_family(m)})),
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# OpenRouter API client
# ---------------------------------------------------------------------------

# Tier 2 — Dev filter: cheap, fast, orthogonal to generation models (DeepSeek/Qwen/Llama).
# Used for bulk pointwise filtering and pairwise tournaments during dataset authoring.
DEV_JUDGE_MODEL = "google/gemini-2.0-flash-exp"

# Tier 3 — Calibration / spot-check: mid-tier, different family from Tier 2.
# Used in --calibrate mode to validate that the dev-tier judge is not systematically
# biased. Scores ~10% of filtered tasks and computes agreement with stored dev scores.
EVAL_JUDGE_MODEL = "anthropic/claude-haiku-20240307"

# Backward-compatible alias — used as the default for --judge-model.
DEFAULT_JUDGE_MODEL = DEV_JUDGE_MODEL


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
                task["metadata"]["judge_tier"] = "dev"
                task["metadata"]["judge_score"] = score
                task["metadata"]["judge_notes"] = judgment.get("notes", "")
                task["metadata"]["judge_criteria"] = {
                    c: judgment.get(c) for c in [
                        "realism", "difficulty_calibration",
                        "ground_truth_quality", "dimension_alignment",
                    ]
                }
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
# Calibration: dev-tier vs eval-tier agreement check
# ---------------------------------------------------------------------------

CRITERIA = ["realism", "difficulty_calibration", "ground_truth_quality", "dimension_alignment"]


def calibrate_with_eval_tier(
    input_dir: Path,
    eval_model: str,
    dev_model: str,
    sample_n: int,
    threshold: int,
    seed: int = 42,
) -> dict:
    """
    Spot-check calibration: compare dev-tier scores (stored in task metadata)
    against eval-tier scores (computed fresh) on a random sample of tasks.

    This validates that the cheap dev-tier judge (Tier 2) is not systematically
    biased relative to the mid-tier eval judge (Tier 3) before the held-out
    evaluation runs.

    Algorithm:
      1. Load all tasks from input_dir that have metadata.judge_score set
         (i.e., tasks that have already been through the dev-tier filter).
      2. Sample min(sample_n, len(tasks)) tasks stratified by dimension.
      3. Re-score each sampled task with the eval-tier model.
      4. Compute per-criterion mean absolute error (MAE) between dev and eval scores.
      5. Compute pass/fail flip rate: fraction of tasks where dev pass != eval pass.
      6. Flag any criterion where MAE > 0.5 (systematic bias threshold).
      7. Return a calibration report dict and print a human-readable summary.

    Args:
        input_dir:   Directory of tasks that have already been dev-tier filtered.
                     Tasks must have metadata.judge_score and per-criterion scores
                     stored by filter_tasks() (metadata.judge_criteria).
        eval_model:  OpenRouter model ID for the eval-tier judge (Tier 3).
        dev_model:   OpenRouter model ID used for the original dev-tier filter.
                     Used for documentation only — not called again.
        sample_n:    Number of tasks to sample. Stratified by dimension.
        threshold:   Pass threshold (same as used in the original filter run).
        seed:        Random seed for reproducible sampling.

    Returns a dict with:
        sampled:          number of tasks actually scored
        dev_model:        model used for original dev-tier scoring
        eval_model:       model used for eval-tier re-scoring
        per_criterion:    dict of criterion -> {mae, dev_mean, eval_mean, bias_flag}
        overall_mae:      mean MAE across all four criteria
        flip_rate:        fraction of tasks where pass/fail verdict changed
        flipped_tasks:    list of task_ids where verdict flipped
        bias_detected:    True if any criterion MAE > 0.5
        recommendation:   "PROCEED" or "INVESTIGATE" with explanation
    """
    if not input_dir.exists():
        print(f"ERROR: input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    task_files = sorted(input_dir.glob("*.json"))
    if not task_files:
        print(f"WARNING: no JSON files found in {input_dir}", file=sys.stderr)
        return {"sampled": 0, "error": "no tasks found"}

    # Load tasks that have dev-tier scores in metadata
    scored_tasks = []
    for tf in task_files:
        try:
            task = json.loads(tf.read_text(encoding="utf-8"))
            meta = task.get("metadata", {})
            # Accept tasks that have judge_score (set by filter_tasks)
            if meta.get("judge_score") is not None:
                scored_tasks.append((tf, task))
        except json.JSONDecodeError:
            continue

    if not scored_tasks:
        print(
            "WARNING: no tasks with metadata.judge_score found. "
            "Run pointwise filter first (filter_tasks writes judge_score to metadata).",
            file=sys.stderr,
        )
        return {"sampled": 0, "error": "no scored tasks found"}

    # Stratified sample by dimension
    by_dim: dict[str, list] = {}
    for tf, task in scored_tasks:
        dim = task.get("dimension", "unknown")
        by_dim.setdefault(dim, []).append((tf, task))

    rng = random.Random(seed)
    sampled: list[tuple[Path, dict]] = []
    dims = sorted(by_dim.keys())
    per_dim_quota = max(1, sample_n // max(len(dims), 1))

    for dim in dims:
        pool = by_dim[dim]
        rng.shuffle(pool)
        sampled.extend(pool[:per_dim_quota])

    # Top up to sample_n if quotas left room
    remaining = [item for item in scored_tasks if item not in sampled]
    rng.shuffle(remaining)
    sampled.extend(remaining[: max(0, sample_n - len(sampled))])
    sampled = sampled[:sample_n]

    print(f"Calibration: {len(sampled)} tasks sampled from {len(scored_tasks)} scored tasks")
    print(f"  Dev-tier model:  {dev_model}")
    print(f"  Eval-tier model: {eval_model}")
    print(f"  Pass threshold:  {threshold}/10")
    print()

    # Re-score with eval-tier model
    dev_scores: list[dict] = []
    eval_scores: list[dict] = []
    flipped_tasks: list[str] = []
    errors = 0

    for i, (tf, task) in enumerate(sampled, 1):
        task_id = task.get("task_id", tf.stem)
        meta = task.get("metadata", {})
        dev_total = meta.get("judge_score", 0)

        # Reconstruct dev per-criterion scores from metadata if available,
        # otherwise treat the stored total as the only signal.
        dev_criteria = meta.get("judge_criteria", {})

        print(f"  [{i}/{len(sampled)}] {task_id} (dev={dev_total}/10)...", end=" ")

        eval_judgment = _judge_task(task, eval_model, threshold)
        if not eval_judgment:
            print("ERROR (eval judge unavailable)")
            errors += 1
            continue

        eval_total = eval_judgment.get("total", 0)
        eval_pass = eval_judgment.get("pass", eval_total >= threshold)
        dev_pass = dev_total >= threshold

        flipped = dev_pass != eval_pass
        if flipped:
            flipped_tasks.append(task_id)
            flip_marker = " *** FLIP ***"
        else:
            flip_marker = ""

        print(f"eval={eval_total}/10  {'PASS' if eval_pass else 'FAIL'}{flip_marker}")

        dev_scores.append({
            "task_id": task_id,
            "total": dev_total,
            "pass": dev_pass,
            **{c: dev_criteria.get(c) for c in CRITERIA},
        })
        eval_scores.append({
            "task_id": task_id,
            "total": eval_total,
            "pass": eval_pass,
            **{c: eval_judgment.get(c) for c in CRITERIA},
        })

    if not eval_scores:
        return {"sampled": 0, "errors": errors, "error": "all eval judge calls failed"}

    # Compute per-criterion MAE
    per_criterion: dict[str, dict] = {}
    for c in CRITERIA:
        dev_vals = [s[c] for s in dev_scores if s.get(c) is not None]
        eval_vals = [s[c] for s in eval_scores if s.get(c) is not None]
        if not dev_vals or not eval_vals:
            per_criterion[c] = {"mae": None, "dev_mean": None, "eval_mean": None, "bias_flag": False}
            continue
        n = min(len(dev_vals), len(eval_vals))
        mae = sum(abs(d - e) for d, e in zip(dev_vals[:n], eval_vals[:n])) / n
        dev_mean = sum(dev_vals[:n]) / n
        eval_mean = sum(eval_vals[:n]) / n
        bias_flag = mae > 0.5
        per_criterion[c] = {
            "mae": round(mae, 3),
            "dev_mean": round(dev_mean, 3),
            "eval_mean": round(eval_mean, 3),
            "bias_flag": bias_flag,
        }

    # Overall MAE (total score, 0–10)
    dev_totals = [s["total"] for s in dev_scores]
    eval_totals = [s["total"] for s in eval_scores]
    n = min(len(dev_totals), len(eval_totals))
    overall_mae = sum(abs(d - e) for d, e in zip(dev_totals[:n], eval_totals[:n])) / max(n, 1)

    flip_rate = len(flipped_tasks) / max(len(eval_scores), 1)
    bias_detected = any(v["bias_flag"] for v in per_criterion.values() if v["mae"] is not None)

    if bias_detected or flip_rate > 0.15:
        recommendation = (
            "INVESTIGATE — dev-tier judge shows systematic bias on one or more criteria "
            f"(MAE > 0.5) or high flip rate ({flip_rate:.1%} > 15%). "
            "Consider re-running the filter with the eval-tier model or raising the threshold."
        )
    else:
        recommendation = (
            f"PROCEED — dev-tier judge is well-calibrated (overall MAE={overall_mae:.2f}, "
            f"flip rate={flip_rate:.1%}). No systematic bias detected."
        )

    result = {
        "sampled": len(eval_scores),
        "errors": errors,
        "dev_model": dev_model,
        "eval_model": eval_model,
        "threshold": threshold,
        "seed": seed,
        "per_criterion": per_criterion,
        "overall_mae": round(overall_mae, 3),
        "flip_rate": round(flip_rate, 4),
        "flipped_tasks": flipped_tasks,
        "bias_detected": bias_detected,
        "recommendation": recommendation,
    }

    # Human-readable summary
    print()
    print(f"{'='*65}")
    print("Calibration Report — Dev-tier vs Eval-tier Agreement")
    print(f"{'='*65}")
    print(f"  Sampled tasks:   {len(eval_scores)} / {len(scored_tasks)}")
    print(f"  Dev model:       {dev_model}")
    print(f"  Eval model:      {eval_model}")
    print(f"  Overall MAE:     {overall_mae:.2f} (total score, 0–10)")
    print(f"  Flip rate:       {flip_rate:.1%} ({len(flipped_tasks)} tasks changed verdict)")
    print()
    print(f"  {'Criterion':<30} {'Dev mean':>9} {'Eval mean':>10} {'MAE':>6} {'Bias?':>6}")
    print(f"  {'-'*30} {'-'*9} {'-'*10} {'-'*6} {'-'*6}")
    for c, v in per_criterion.items():
        if v["mae"] is None:
            print(f"  {c:<30} {'N/A':>9} {'N/A':>10} {'N/A':>6} {'N/A':>6}")
        else:
            flag = "YES ⚠" if v["bias_flag"] else "no"
            print(f"  {c:<30} {v['dev_mean']:>9.2f} {v['eval_mean']:>10.2f} "
                  f"{v['mae']:>6.3f} {flag:>6}")
    print()
    if flipped_tasks:
        print(f"  Flipped tasks ({len(flipped_tasks)}):")
        for tid in flipped_tasks[:10]:
            print(f"    {tid}")
        if len(flipped_tasks) > 10:
            print(f"    ... and {len(flipped_tasks) - 10} more")
        print()
    print(f"  Recommendation: {recommendation}")
    print(f"{'='*65}")

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
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help=(
            "Run calibration mode: re-score a sample of dev-tier-filtered tasks "
            "with the eval-tier model and report agreement / bias."
        ),
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

    # Calibration-specific args
    parser.add_argument(
        "--eval-model",
        default=EVAL_JUDGE_MODEL,
        help=(
            f"Eval-tier model for --calibrate spot-check (default: {EVAL_JUDGE_MODEL}). "
            "Must be a different family from --judge-model."
        ),
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=25,
        help="Number of tasks to sample in --calibrate mode (default: 25, ~10%% of 250)",
    )
    parser.add_argument(
        "--calibrate-seed",
        type=int,
        default=42,
        help="Random seed for calibration sampling (default: 42)",
    )

    # Shared args
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help=(
            f"Dev-tier judge model for pointwise/pairwise filtering "
            f"(default: {DEFAULT_JUDGE_MODEL}). "
            "Must be a different family from the generation models."
        ),
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

    # ── Anti-leakage guard — must run before any API calls ─────────────────
    # Fails hard if any judge/eval model shares a family with a generation model.
    # This enforces the Li et al. (2025) invariant: Tier 1 ∉ {Tier 2, Tier 3}.
    assert_no_family_overlap(
        judge_model=args.judge_model,
        eval_model=args.eval_model if args.calibrate else None,
    )

    # ── Calibration mode ───────────────────────────────────────────────────
    if args.calibrate:
        if not args.input_dir:
            print("ERROR: --input-dir required for --calibrate mode", file=sys.stderr)
            sys.exit(1)

        if args.judge_model == args.eval_model:
            print(
                f"WARNING: --judge-model and --eval-model are the same ({args.judge_model}). "
                "Calibration requires two different model families to be meaningful. "
                "(The family-separation check above should have caught this — "
                "please add both models to MODEL_FAMILIES if they are missing.)",
                file=sys.stderr,
            )

        result = calibrate_with_eval_tier(
            input_dir=args.input_dir,
            eval_model=args.eval_model,
            dev_model=args.judge_model,
            sample_n=args.sample_n,
            threshold=args.threshold,
            seed=args.calibrate_seed,
        )

        if args.json:
            print(json.dumps(result, indent=2))

        sys.exit(0 if not result.get("bias_detected") else 1)

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
        print(f"Judge model (Tier 2 — dev): {args.judge_model}")
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
    print(f"Judge model (Tier 2 — dev): {args.judge_model}")
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
    print()
    print(f"Next step: run calibration to validate dev-tier judge quality:")
    print(f"  python generation_scripts/judge_filter.py \\")
    print(f"    --calibrate \\")
    print(f"    --input-dir {args.output_dir or args.input_dir} \\")
    print(f"    --sample-n {max(10, summary['passed'] // 10)} \\")
    print(f"    --eval-model {EVAL_JUDGE_MODEL}")

    if args.json:
        print()
        print(json.dumps(summary, indent=2))

    sys.exit(0 if summary["errors"] == 0 else 1)


if __name__ == "__main__":
    main()
