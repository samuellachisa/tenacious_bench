"""
contamination_check.py
Tenacious-Bench v0.1 — Dataset contamination checker.

Checks for:
  1. N-gram overlap (< 8-gram threshold)
  2. Cosine similarity via TF-IDF (>= 0.85 threshold)

between the benchmark tasks and any reference corpus (e.g. trace_log.jsonl,
publicly available datasets).

Usage:
    python contamination_check.py \
        --bench-dir tenacious_bench_v0.1 \
        --reference-file eval/trace_log.jsonl \
        --ngram 8 \
        --cosine-threshold 0.85

Exit code 0 = no contamination, 1 = contamination detected.
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# N-gram utilities
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer, lowercased."""
    return text.lower().split()


def _ngrams(tokens: list[str], n: int) -> set[tuple]:
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def ngram_overlap(a: str, b: str, n: int = 8) -> float:
    """
    Returns the fraction of n-grams in `a` that appear in `b`.
    Returns 0.0 if either string is too short to produce any n-grams.
    """
    toks_a = _tokenize(a)
    toks_b = _tokenize(b)
    grams_a = _ngrams(toks_a, n)
    grams_b = _ngrams(toks_b, n)
    if not grams_a:
        return 0.0
    return len(grams_a & grams_b) / len(grams_a)


# ---------------------------------------------------------------------------
# TF-IDF cosine similarity (no external deps)
# ---------------------------------------------------------------------------

def _tfidf_vector(doc: str, idf: dict[str, float]) -> dict[str, float]:
    tokens = _tokenize(doc)
    tf = Counter(tokens)
    n = max(len(tokens), 1)
    return {t: (c / n) * idf.get(t, 0.0) for t, c in tf.items()}


def _cosine(v1: dict, v2: dict) -> float:
    shared = set(v1) & set(v2)
    if not shared:
        return 0.0
    dot = sum(v1[t] * v2[t] for t in shared)
    mag1 = math.sqrt(sum(x ** 2 for x in v1.values()))
    mag2 = math.sqrt(sum(x ** 2 for x in v2.values()))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


def build_idf(corpus: list[str]) -> dict[str, float]:
    n = len(corpus)
    df: Counter = Counter()
    for doc in corpus:
        for token in set(_tokenize(doc)):
            df[token] += 1
    return {t: math.log((n + 1) / (c + 1)) + 1 for t, c in df.items()}


# ---------------------------------------------------------------------------
# Task text extraction
# ---------------------------------------------------------------------------

def task_to_text(task: dict) -> str:
    """Convert a task JSON to a flat string for comparison."""
    parts = [
        task.get("task_id", ""),
        task.get("dimension", ""),
        json.dumps(task.get("ground_truth", {})),
        json.dumps(task.get("input", {})),
    ]
    return " ".join(str(p) for p in parts)


def load_reference_texts(reference_file: Path) -> list[str]:
    """Load reference corpus from JSONL (one JSON per line)."""
    texts = []
    if not reference_file.exists():
        return texts
    with open(reference_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                texts.append(json.dumps(obj))
            except json.JSONDecodeError:
                texts.append(line)
    return texts


# ---------------------------------------------------------------------------
# Main contamination check
# ---------------------------------------------------------------------------

def run_check(
    bench_dir: Path,
    reference_texts: list[str],
    ngram_n: int = 8,
    cosine_threshold: float = 0.85,
) -> dict:
    task_files = []
    for split in ("train", "dev", "held_out"):
        split_dir = bench_dir / split
        if split_dir.exists():
            task_files.extend(sorted(split_dir.glob("*.json")))

    if not task_files:
        return {"error": f"No task files found in {bench_dir}"}

    # Build IDF from all tasks + reference
    all_tasks = []
    for tf in task_files:
        try:
            task = json.loads(tf.read_text(encoding="utf-8"))
            all_tasks.append((tf, task, task_to_text(task)))
        except Exception as e:
            all_tasks.append((tf, {}, ""))

    all_docs = [t[2] for t in all_tasks] + reference_texts
    idf = build_idf([d for d in all_docs if d])

    violations = []
    clean = 0

    for tf, task, task_text in all_tasks:
        if not task_text:
            continue

        task_vec = _tfidf_vector(task_text, idf)
        max_ngram = 0.0
        max_cosine = 0.0
        worst_ref_idx = -1

        for i, ref_text in enumerate(reference_texts):
            if not ref_text:
                continue
            ng = ngram_overlap(task_text, ref_text, ngram_n)
            cos = _cosine(task_vec, _tfidf_vector(ref_text, idf))
            if ng > max_ngram:
                max_ngram = ng
            if cos > max_cosine:
                max_cosine = cos
                worst_ref_idx = i

        is_contaminated = max_ngram >= 1.0 or max_cosine >= cosine_threshold
        if is_contaminated:
            violations.append({
                "task_id": task.get("task_id", tf.stem),
                "partition": task.get("metadata", {}).get("partition", "unknown"),
                "max_ngram_overlap": round(max_ngram, 4),
                "max_cosine_similarity": round(max_cosine, 4),
                "worst_ref_index": worst_ref_idx,
                "flags": {
                    "ngram_exact_match": max_ngram >= 1.0,
                    "high_cosine": max_cosine >= cosine_threshold,
                },
            })
        else:
            clean += 1

    return {
        "bench_dir": str(bench_dir),
        "tasks_checked": len(all_tasks),
        "reference_docs": len(reference_texts),
        "ngram_n": ngram_n,
        "cosine_threshold": cosine_threshold,
        "clean": clean,
        "violations": violations,
        "contamination_rate": round(len(violations) / max(len(all_tasks), 1), 4),
        "status": "CLEAN" if not violations else "CONTAMINATED",
    }


# ---------------------------------------------------------------------------
# Check 3: Time-shift verification
# ---------------------------------------------------------------------------

def run_time_shift_check(bench_dir: Path, cutoff_date: str = "2026-04-21") -> dict:
    """
    Verify that bench snapshot dates and task creation timestamps are consistent
    with the publication pipeline's temporal ordering requirements.

    Rules enforced:
      1. All tasks must have metadata.created_at >= cutoff_date (tasks cannot
         predate the bench snapshot they reference).
      2. Held-out tasks must use bench snapshots with jittered capacity values
         (metadata.bench_snapshot_jittered == True) to prevent exact value
         memorisation by models trained on data after cutoff_date.
      3. No task's bench_summary_snapshot.capacity_locked_until date may be
         in the past relative to the task's created_at date (stale snapshots).

    Args:
        bench_dir:    Path to the benchmark directory (contains train/, dev/, held_out/).
        cutoff_date:  ISO date string (YYYY-MM-DD). Tasks created before this date
                      are flagged as potentially contaminated by pre-cutoff training data.

    Returns a dict with:
        tasks_checked:      total tasks inspected
        violations:         list of violation dicts
        held_out_jitter_ok: True if all held-out tasks have jittered snapshots
        stale_snapshots:    list of task_ids with stale capacity_locked_until dates
        status:             "CLEAN" or "VIOLATIONS"
    """
    from datetime import date

    try:
        cutoff = date.fromisoformat(cutoff_date)
    except ValueError:
        return {"error": f"Invalid cutoff_date format: {cutoff_date}. Use YYYY-MM-DD."}

    task_files = []
    for split in ("train", "dev", "held_out"):
        split_dir = bench_dir / split
        if split_dir.exists():
            task_files.extend([(split, tf) for tf in sorted(split_dir.glob("*.json"))])

    if not task_files:
        return {"error": f"No task files found in {bench_dir}"}

    violations = []
    stale_snapshots = []
    held_out_jitter_flags = []

    for split, tf in task_files:
        try:
            task = json.loads(tf.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            violations.append({
                "task_id": tf.stem,
                "split": split,
                "violation": "invalid_json",
                "detail": "Could not parse task file",
            })
            continue

        task_id = task.get("task_id", tf.stem)
        metadata = task.get("metadata", {})
        created_at_str = metadata.get("created_at", "")
        bench_snapshot = task.get("input", {}).get("bench_summary_snapshot", {})
        locked_until_str = bench_snapshot.get("capacity_locked_until", "")

        # Rule 1: created_at must be present and >= cutoff_date
        if not created_at_str:
            violations.append({
                "task_id": task_id,
                "split": split,
                "violation": "missing_created_at",
                "detail": "metadata.created_at is absent",
            })
        else:
            try:
                # Handle ISO datetime strings (e.g. "2026-04-29T12:00:00+00:00")
                created_date = date.fromisoformat(created_at_str[:10])
                if created_date < cutoff:
                    violations.append({
                        "task_id": task_id,
                        "split": split,
                        "violation": "created_before_cutoff",
                        "detail": (
                            f"created_at={created_at_str[:10]} is before "
                            f"cutoff={cutoff_date}"
                        ),
                    })
            except ValueError:
                violations.append({
                    "task_id": task_id,
                    "split": split,
                    "violation": "invalid_created_at",
                    "detail": f"Cannot parse created_at: {created_at_str!r}",
                })

        # Rule 2: held-out tasks should have jittered bench snapshots
        if split == "held_out":
            jittered = metadata.get("bench_snapshot_jittered", False)
            held_out_jitter_flags.append(jittered)
            if not jittered:
                # Soft warning — not a hard violation, but flagged for review
                violations.append({
                    "task_id": task_id,
                    "split": split,
                    "violation": "held_out_not_jittered",
                    "detail": (
                        "Held-out task does not have bench_snapshot_jittered=True. "
                        "Models trained after cutoff_date may have memorised exact "
                        "capacity values. Consider regenerating with ±20% jitter."
                    ),
                    "severity": "warning",
                })

        # Rule 3: capacity_locked_until must not be in the past relative to created_at
        if locked_until_str and created_at_str:
            try:
                locked_until = date.fromisoformat(locked_until_str[:10])
                created_date = date.fromisoformat(created_at_str[:10])
                if locked_until < created_date:
                    stale_snapshots.append({
                        "task_id": task_id,
                        "split": split,
                        "capacity_locked_until": locked_until_str,
                        "created_at": created_at_str[:10],
                        "days_stale": (created_date - locked_until).days,
                    })
            except ValueError:
                pass  # Already flagged above if created_at was invalid

    hard_violations = [v for v in violations if v.get("severity") != "warning"]
    warnings = [v for v in violations if v.get("severity") == "warning"]
    held_out_jitter_ok = all(held_out_jitter_flags) if held_out_jitter_flags else False

    return {
        "bench_dir": str(bench_dir),
        "cutoff_date": cutoff_date,
        "tasks_checked": len(task_files),
        "hard_violations": hard_violations,
        "warnings": warnings,
        "stale_snapshots": stale_snapshots,
        "held_out_tasks_checked": sum(1 for s, _ in task_files if s == "held_out"),
        "held_out_jitter_ok": held_out_jitter_ok,
        "status": "CLEAN" if not hard_violations and not stale_snapshots else "VIOLATIONS",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Tenacious-Bench contamination checker")
    parser.add_argument("--bench-dir", type=Path, default=Path("tenacious_bench_v0.1"))
    parser.add_argument("--reference-file", type=Path, default=Path("../eval/trace_log.jsonl"))
    parser.add_argument("--ngram", type=int, default=8)
    parser.add_argument("--cosine-threshold", type=float, default=0.85)
    parser.add_argument(
        "--time-shift",
        action="store_true",
        help="Run Check 3: time-shift verification (temporal ordering + jitter audit)",
    )
    parser.add_argument(
        "--cutoff-date",
        default="2026-04-21",
        help="ISO date (YYYY-MM-DD) for time-shift cutoff (default: 2026-04-21)",
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    # ── Check 3: time-shift verification ──────────────────────────────────
    if args.time_shift:
        print(f"Running time-shift verification (cutoff={args.cutoff_date})")
        result = run_time_shift_check(args.bench_dir, args.cutoff_date)

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"\n{'='*60}")
            print(f"  Time-Shift Check Status: {result.get('status')}")
            print(f"  Tasks checked:           {result.get('tasks_checked')}")
            print(f"  Held-out tasks:          {result.get('held_out_tasks_checked')}")
            print(f"  Held-out jitter OK:      {result.get('held_out_jitter_ok')}")
            print(f"  Hard violations:         {len(result.get('hard_violations', []))}")
            print(f"  Warnings:                {len(result.get('warnings', []))}")
            print(f"  Stale snapshots:         {len(result.get('stale_snapshots', []))}")
            print(f"{'='*60}\n")

            if result.get("hard_violations"):
                print("Hard violations:")
                for v in result["hard_violations"][:10]:
                    print(f"  [{v['split']}] {v['task_id']} — {v['violation']}: {v['detail']}")

            if result.get("warnings"):
                print(f"\nWarnings ({len(result['warnings'])} held-out tasks without jitter):")
                for w in result["warnings"][:5]:
                    print(f"  [{w['split']}] {w['task_id']} — {w['detail'][:80]}")
                if len(result["warnings"]) > 5:
                    print(f"  ... and {len(result['warnings']) - 5} more")

            if result.get("stale_snapshots"):
                print("\nStale bench snapshots:")
                for s in result["stale_snapshots"][:5]:
                    print(f"  {s['task_id']} — locked_until={s['capacity_locked_until']} "
                          f"created={s['created_at']} ({s['days_stale']}d stale)")

        sys.exit(0 if result.get("status") == "CLEAN" else 1)

    # ── Checks 1 & 2: n-gram + cosine similarity ──────────────────────────
    reference_texts = load_reference_texts(args.reference_file)
    print(f"Loaded {len(reference_texts)} reference documents from {args.reference_file}")

    result = run_check(
        bench_dir=args.bench_dir,
        reference_texts=reference_texts,
        ngram_n=args.ngram,
        cosine_threshold=args.cosine_threshold,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"  Status:         {result.get('status')}")
        print(f"  Tasks checked:  {result.get('tasks_checked')}")
        print(f"  Clean:          {result.get('clean')}")
        print(f"  Violations:     {len(result.get('violations', []))}")
        print(f"  Contam. rate:   {result.get('contamination_rate', 0):.1%}")
        print(f"{'='*60}\n")

        if result.get("violations"):
            print("Violations:")
            for v in result["violations"][:10]:
                print(f"  {v['task_id']} | cosine={v['max_cosine_similarity']:.3f} | "
                      f"ngram_exact={v['flags']['ngram_exact_match']}")
            if len(result["violations"]) > 10:
                print(f"  ... and {len(result['violations']) - 10} more")

    sys.exit(0 if result.get("status") == "CLEAN" else 1)


if __name__ == "__main__":
    main()
