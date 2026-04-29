"""
generation_scripts/judge_filter.py
Tenacious-Bench v0.1 — Post-generation quality filter.

Applies a judge model to filter generated tasks before inclusion in the benchmark.
This is a standalone filter that can be applied to any task directory.

Judge model must be a different family from the generator models to prevent
preference leakage (Li et al. 2025).

Usage:
    # Filter a directory of tasks
    python generation_scripts/judge_filter.py \
        --input-dir tenacious_bench_v0.1/train \
        --output-dir tenacious_bench_v0.1/train_filtered \
        --judge-model google/gemini-2.0-flash-exp \
        --threshold 7

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
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Tenacious-Bench v0.1 — Judge filter for generated tasks"
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory of tasks to filter")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to copy passing tasks (if omitted, no files are moved)",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help=f"OpenRouter model ID for judge (default: {DEFAULT_JUDGE_MODEL})",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=7,
        help="Minimum score (0–10) to pass (default: 7)",
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

    if args.dry_run:
        print(f"DRY RUN: no files will be moved")

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
    print("Summary")
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

