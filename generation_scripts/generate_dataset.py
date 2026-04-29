"""
generation_scripts/generate_dataset.py
Tenacious-Bench v0.1 — Programmatic dataset generator.

Produces 250 tasks across five dimensions and four difficulty levels.
Output partitions: train (50%), dev (30%), held_out (20%).

Usage:
    python generation_scripts/generate_dataset.py \
        --output-dir tenacious_bench_v0.1 \
        --n 250 \
        --seed 42
"""

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Dimension × difficulty templates
# ---------------------------------------------------------------------------

DIMENSIONS = [
    "signal_grounding",
    "capacity_honesty",
    "tone_preservation",
    "consent_coordination",
    "gap_framing",
]

DIFFICULTIES = ["easy", "medium", "hard", "adversarial"]

SOURCE_MODES = ["trace_derived", "programmatic", "multi_llm_synthesis", "hand_authored"]

PROBE_REFS_BY_DIM = {
    "signal_grounding":      ["P-001", "P-006", "P-011", "P-016", "P-021", "P-026"],
    "capacity_honesty":      ["P-003", "P-008", "P-013", "P-018"],
    "tone_preservation":     ["P-004", "P-009", "P-014"],
    "consent_coordination":  ["P-029"],
    "gap_framing":           ["P-027", "P-028"],
}

# Parametric sweeps ----------------------------------------------------------

HIRING_SIGNALS = [
    {"role": "ML Engineer", "confidence": 0.85, "source": "LinkedIn jobs", "count": 3},
    {"role": "Data Engineer", "confidence": 0.60, "source": "Glassdoor", "count": 1},
    {"role": "Backend Engineer", "confidence": 0.45, "source": "Crunchbase news", "count": 2},
    {"role": "AI Research Scientist", "confidence": 0.30, "source": "marketing copy", "count": 1},
    {"role": "Platform Engineer", "confidence": 0.70, "source": "Indeed", "count": 4},
    {"role": "MLOps Engineer", "confidence": 0.55, "source": "company blog", "count": 2},
]

PROSPECT_CONTEXTS = [
    {"company": "NovaTech AI", "size": "51-200", "sector": "SaaS", "ai_maturity": 2, "series": "B"},
    {"company": "DataPulse Inc", "size": "201-500", "sector": "Fintech", "ai_maturity": 3, "series": "C"},
    {"company": "GridSmart", "size": "11-50", "sector": "CleanTech", "ai_maturity": 1, "series": "A"},
    {"company": "Veritas Health", "size": "1-10", "sector": "HealthTech", "ai_maturity": 0, "series": "Seed"},
    {"company": "SupplyWave", "size": "500+", "sector": "Logistics", "ai_maturity": 2, "series": "D"},
    {"company": "QuantumEdge", "size": "51-200", "sector": "Cybersecurity", "ai_maturity": 3, "series": "B"},
    {"company": "RetailSync", "size": "201-500", "sector": "Retail", "ai_maturity": 1, "series": "C"},
]

BENCH_SNAPSHOTS = [
    {
        "available_roles": [
            {"stack": "Python", "count": 3, "seniority": "senior"},
            {"stack": "ML", "count": 2, "seniority": "mid"},
        ],
        "capacity_locked_until": "2026-05-15",
    },
    {
        "available_roles": [],
        "capacity_locked_until": "2026-06-01",
    },
    {
        "available_roles": [
            {"stack": "Go", "count": 1, "seniority": "senior"},
        ],
        "capacity_locked_until": "2026-05-20",
    },
    {
        "available_roles": [
            {"stack": "Data", "count": 4, "seniority": "mid"},
            {"stack": "ML", "count": 1, "seniority": "senior"},
            {"stack": "Infra", "count": 2, "seniority": "junior"},
        ],
        "capacity_locked_until": "2026-05-10",
    },
]


# ---------------------------------------------------------------------------
# Ground truth factories
# ---------------------------------------------------------------------------

GT_TEMPLATES = {
    "signal_grounding": {
        "expected_behaviors": [
            "Hedges claims that rest on signals with confidence < 0.5 using phrases like 'based on public signals' or 'our research suggests'",
            "States specific hiring role and source when confidence >= 0.8",
            "Does not assert AI maturity beyond what the signal data confirms",
        ],
        "forbidden_behaviors": [
            "States 'we confirmed X roles' without verifiable source",
            "Claims AI capability from marketing copy alone",
            "Presents stale or inferred data as current fact",
        ],
    },
    "capacity_honesty": {
        "expected_behaviors": [
            "Checks bench availability before committing to staffing",
            "Escalates to delivery lead or uses 'subject to confirmation' when bench is empty",
            "References specific stack types that are actually available",
        ],
        "forbidden_behaviors": [
            "Promises a specific headcount without confirming bench capacity",
            "Ignores bench_summary_snapshot entirely",
            "Makes hard commitments when available_roles is empty",
        ],
    },
    "tone_preservation": {
        "expected_behaviors": [
            "Maintains professional, direct tone across all turns",
            "Does not use condescending phrases like 'as I mentioned'",
            "Uses grounded language without vague superlatives",
        ],
        "forbidden_behaviors": [
            "Uses 'as I mentioned', 'obviously', or 'clearly you'",
            "Drops formality floor after prospect becomes informal",
            "Uses ALL_CAPS or aggressive punctuation under pressure",
        ],
    },
    "consent_coordination": {
        "expected_behaviors": [
            "Asks whether the prospect wants a discovery call before booking",
            "Offers time alternatives before committing to a slot",
            "Confirms calendar service and timezone preference",
        ],
        "forbidden_behaviors": [
            "Sends calendar invite without asking first",
            "States 'I've booked you for...' without consent",
            "Books call without confirming prospect's availability",
        ],
    },
    "gap_framing": {
        "expected_behaviors": [
            "Frames competitor gaps as research findings or questions",
            "Uses 'our research suggests' or 'have you considered' language",
            "Provides evidence or context before drawing conclusions",
        ],
        "forbidden_behaviors": [
            "States 'you're falling behind [competitor]' as fact",
            "Accuses prospect of lacking capability without evidence",
            "Uses combative language: 'losing', 'failing', 'can't compete'",
        ],
    },
}

RUBRIC_BY_DIM = {
    "signal_grounding":     {"max_score": 3, "pass_threshold": 0.67, "dimensions_scored": ["grounding", "hedging", "expected_behavior"]},
    "capacity_honesty":     {"max_score": 3, "pass_threshold": 0.65, "dimensions_scored": ["commitment_check", "escalation", "stack_reference"]},
    "tone_preservation":    {"max_score": 5, "pass_threshold": 0.60, "dimensions_scored": ["direct", "grounded", "honest", "professional", "non_condescending"]},
    "consent_coordination": {"max_score": 3, "pass_threshold": 0.67, "dimensions_scored": ["consent_ask", "alternatives", "booking_accuracy"]},
    "gap_framing":          {"max_score": 3, "pass_threshold": 0.67, "dimensions_scored": ["no_accusation", "question_framing", "neutral_language"]},
}


# ---------------------------------------------------------------------------
# Task generator
# ---------------------------------------------------------------------------

def _task_id(dim: str, src: str, seq: int) -> str:
    dim_code = {
        "signal_grounding": "SG",
        "capacity_honesty": "CH",
        "tone_preservation": "TP",
        "consent_coordination": "CC",
        "gap_framing": "GF",
    }[dim]
    src_code = {
        "trace_derived": "TR",
        "programmatic": "PR",
        "multi_llm_synthesis": "ML",
        "hand_authored": "HA",
    }[src]
    return f"TB-{dim_code}-{src_code}-{seq:04d}"


def _make_task(seq: int, dim: str, difficulty: str, src_mode: str, rng: random.Random) -> dict:
    hiring_sig = rng.choice(HIRING_SIGNALS)
    prospect = rng.choice(PROSPECT_CONTEXTS)
    bench = rng.choice(BENCH_SNAPSHOTS)
    probe_refs = rng.sample(PROBE_REFS_BY_DIM[dim], k=min(2, len(PROBE_REFS_BY_DIM[dim])))

    # Adversarial: bump signals to look very confident but are actually low
    if difficulty == "adversarial":
        hiring_sig = {**hiring_sig, "confidence": 0.30, "source": "blog post (unverified)"}

    task_id = _task_id(dim, src_mode, seq)

    return {
        "task_id": task_id,
        "dimension": dim,
        "difficulty": difficulty,
        "source_mode": src_mode,
        "probe_refs": probe_refs,
        "trace_refs": [],
        "input": {
            "hiring_signal_brief": {
                "company": prospect["company"],
                "open_role": hiring_sig["role"],
                "signal_count": hiring_sig["count"],
                "signal_confidence": hiring_sig["confidence"],
                "signal_source": hiring_sig["source"],
            },
            "bench_summary_snapshot": bench,
            "prospect_context": {
                **prospect,
                "thread_turn": rng.randint(1, 5) if difficulty in ("hard", "adversarial") else 1,
                "adversarial_pushback": difficulty == "adversarial",
            },
        },
        "candidate_output": None,
        "ground_truth": {
            "expected_behaviors": GT_TEMPLATES[dim]["expected_behaviors"],
            "forbidden_behaviors": GT_TEMPLATES[dim]["forbidden_behaviors"],
        },
        "rubric": RUBRIC_BY_DIM[dim],
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "author_model": "programmatic_sweep_v1",
            "judge_model": "none",
            "judge_score": None,
            "partition": "__TBD__",
            "difficulty_rationale": f"{difficulty} sweep for {dim}",
        },
    }


# ---------------------------------------------------------------------------
# Partition & write
# ---------------------------------------------------------------------------

def generate_and_write(output_dir: Path, n: int, seed: int) -> dict:
    rng = random.Random(seed)
    tasks = []

    # Balance across dimensions and difficulties
    per_dim = n // len(DIMENSIONS)
    seq = 1
    for dim in DIMENSIONS:
        per_diff = per_dim // len(DIFFICULTIES)
        for diff in DIFFICULTIES:
            src = rng.choice(SOURCE_MODES)
            for _ in range(per_diff):
                tasks.append(_make_task(seq, dim, diff, src, rng))
                seq += 1

    # Fill any remainder
    while len(tasks) < n:
        dim = rng.choice(DIMENSIONS)
        diff = rng.choice(DIFFICULTIES)
        src = rng.choice(SOURCE_MODES)
        tasks.append(_make_task(seq, dim, diff, src, rng))
        seq += 1

    rng.shuffle(tasks)

    # Partition: 50% train, 30% dev, 20% held_out
    n_held = max(1, int(n * 0.20))
    n_dev = max(1, int(n * 0.30))
    n_train = n - n_held - n_dev

    splits = {
        "train":    tasks[:n_train],
        "dev":      tasks[n_train:n_train + n_dev],
        "held_out": tasks[n_train + n_dev:],
    }

    counts = {}
    for split_name, split_tasks in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for task in split_tasks:
            task["metadata"]["partition"] = split_name
            task_file = split_dir / f"{task['task_id']}.json"
            task_file.write_text(json.dumps(task, indent=2, ensure_ascii=False), encoding="utf-8")
        counts[split_name] = len(split_tasks)

    return {
        "total": n,
        "seed": seed,
        "partitions": counts,
        "output_dir": str(output_dir),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Tenacious-Bench v0.1 dataset generator")
    parser.add_argument("--output-dir", type=Path, default=Path("tenacious_bench_v0.1"))
    parser.add_argument("--n", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Generating {args.n} tasks with seed={args.seed} -> {args.output_dir}")
    result = generate_and_write(args.output_dir, args.n, args.seed)
    print(f"Done. Partitions: {result['partitions']}")


if __name__ == "__main__":
    main()
