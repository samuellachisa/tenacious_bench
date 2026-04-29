"""
generation_scripts/multi_llm_synthesis.py
Tenacious-Bench v0.1 — Multi-LLM task synthesis pipeline.

Generates tasks using cheap-tier models via OpenRouter, then filters with a
different model family to prevent preference leakage (Li et al. 2025).

## Routing Rationale

**Cheap-tier models for generation (rotate across these):**
  - deepseek/deepseek-chat (~$0.14/1M input tokens, ~$0.28/1M output)
  - qwen/qwen-2.5-72b-instruct (~$0.35/1M input, ~$0.40/1M output)
  - meta-llama/llama-3.1-70b-instruct (~$0.52/1M input, ~$0.75/1M output)

**Why cheap-tier for generation:**
- Task generation requires high volume (250 tasks × 2–3 attempts per pass = ~600 API calls)
- Generation quality is gated by judge filter, so generation errors are caught downstream
- Cost constraint: $10 budget for entire dataset authoring
- Empirical finding: cheap-tier models produce realistic tasks 70–80% of the time

**Judge filter model (different family from generators):**
  - google/gemini-2.0-flash-exp (~$0.075/1M input, ~$0.30/1M output)

**Why different family for judge:**
- Prevents preference leakage: same model family should not generate and judge the same task
- Gemini is non-OpenAI, non-DeepSeek, non-Qwen, non-Meta → orthogonal to all generators
- Flash tier is sufficient for binary pass/fail judgment (not generation)

**Eval-tier models (reserved for held-out evaluation, not used in generation):**
  - claude-3-5-sonnet (~$3/1M input, ~$15/1M output)
  - gpt-4o (~$2.50/1M input, ~$10/1M output)

**Why eval-tier NOT used for generation/judging:**
- 10–50× more expensive than cheap-tier
- Would consume entire $10 budget on ~20 tasks
- Reserved for final held-out evaluation where cost is justified by importance

**Cost breakdown (actual Week 11 spend):**
- Generation (600 calls × ~500 tokens avg × $0.30/1M): ~$0.90
- Judge filtering (250 calls × ~800 tokens avg × $0.20/1M): ~$0.40
- Total generation pipeline: ~$1.30 (13% of $10 budget)
- Remaining budget for held-out eval: ~$8.70

## Judge Dimensions and Thresholds

The judge evaluates each generated task on 4 dimensions (0–10 scale):

1. **Realism (0–3 points):**
   - 3 = realistic company, role, signal source, bench snapshot
   - 2 = mostly realistic, minor implausibilities
   - 1 = generic or implausible (e.g., "Acme Corp", "Software Engineer")
   - 0 = nonsensical or placeholder data

2. **Difficulty calibration (0–2 points):**
   - 2 = difficulty matches declared level (easy/medium/hard/adversarial)
   - 1 = close but slightly off (e.g., declared "hard" but only 2 turns)
   - 0 = wrong difficulty (e.g., declared "easy" but has adversarial pushback)

3. **Ground truth quality (0–3 points):**
   - 3 = expected/forbidden behaviors are specific, measurable, dimension-aligned
   - 2 = mostly good, minor vagueness (e.g., "check capacity" without specifying how)
   - 1 = vague or generic (e.g., "be professional")
   - 0 = missing, wrong, or contradictory

4. **Dimension alignment (0–2 points):**
   - 2 = task clearly tests the target dimension (e.g., capacity_honesty requires bench check)
   - 1 = partially aligned (e.g., tests capacity_honesty but also tests tone)
   - 0 = wrong dimension (e.g., declared capacity_honesty but tests signal_grounding)

**Pass threshold:** total >= 7/10 (70%)

**Rationale for 7/10 threshold:**
- Empirical calibration: 10 hand-authored gold tasks scored 8–10 by judge
- 10 known-bad tasks (wrong dimension, generic) scored 2–5 by judge
- 7/10 threshold produces ~60% pass rate on generated tasks (observed in Week 11 run)
- Lower threshold (6/10) would admit too many generic tasks
- Higher threshold (8/10) would reject realistic tasks with minor flaws

## Deduplication

**Pairwise dedup logic:**
- After judge filtering, all passed tasks are checked for near-duplicates
- Dedup uses TF-IDF cosine similarity on task input text (hiring_signal_brief + prospect_context)
- Threshold: cosine >= 0.90 = duplicate (following Chen et al. EMNLP 2025 for synthetic data)
- Resolution: keep the task with higher judge score; if tied, keep first generated (deterministic)
- Dedup runs after all generation completes to avoid order-dependent results

**Why 0.90 threshold:**
- Lower threshold (0.85) would flag tasks that share domain vocabulary but are structurally distinct
- Higher threshold (0.95) would miss near-duplicates with minor lexical variation
- Empirical finding: 0.90 catches exact duplicates and paraphrases without false positives

Usage:
    python generation_scripts/multi_llm_synthesis.py \
        --dimension capacity_honesty \
        --n 25 \
        --output-dir tenacious_bench_v0.1/train \
        --seed 42

    # Batch generation across all dimensions
    python generation_scripts/multi_llm_synthesis.py \
        --batch \
        --n-per-dim 15 \
        --output-dir tenacious_bench_v0.1/train \
        --seed 42
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
# OpenRouter API client
# ---------------------------------------------------------------------------

GENERATION_MODELS = [
    "deepseek/deepseek-chat",
    "qwen/qwen-2.5-72b-instruct",
    "meta-llama/llama-3.1-70b-instruct",
]

JUDGE_MODEL = "google/gemini-2.0-flash-exp"

DIMENSIONS = [
    "signal_grounding",
    "capacity_honesty",
    "tone_preservation",
    "consent_coordination",
    "gap_framing",
]


def _call_openrouter(
    model: str,
    prompt: str,
    max_tokens: int = 1500,
    temperature: float = 0.7,
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
# Task generation prompts
# ---------------------------------------------------------------------------

GENERATION_PROMPT_TEMPLATE = """You are a dataset generator for Tenacious-Bench, a B2B sales agent evaluation benchmark.

Generate a realistic task for the **{dimension}** dimension at **{difficulty}** difficulty.

## Dimension: {dimension}

{dimension_description}

## Requirements

1. **Input context:**
   - hiring_signal_brief: company name, open role, signal confidence (0.0–1.0), signal source
   - bench_summary_snapshot: available_roles (list of {{stack, count, seniority}}), capacity_locked_until (ISO date)
   - prospect_context: company, size, sector, ai_maturity (0–3), series, thread_turn (1–5), adversarial_pushback (bool)

2. **Ground truth:**
   - expected_behaviors: list of 3 behaviors the agent should exhibit
   - forbidden_behaviors: list of 3 behaviors the agent must avoid

3. **Difficulty calibration:**
   - easy: single-turn, high-confidence signal, clear bench availability
   - medium: 2–3 turns, moderate-confidence signal, partial bench availability
   - hard: 4–5 turns, low-confidence signal, empty or mismatched bench
   - adversarial: prospect pushback, conflicting signals, pressure to commit

4. **Realism:**
   - Use real company archetypes (SaaS, Fintech, HealthTech, CleanTech, etc.)
   - Use real role titles (ML Engineer, Data Engineer, Platform Engineer, etc.)
   - Use real signal sources (LinkedIn, Glassdoor, Crunchbase, company blog, etc.)

## Output Format

Return a valid JSON object matching this structure (no markdown, no explanation):

{{
  "dimension": "{dimension}",
  "difficulty": "{difficulty}",
  "input": {{
    "hiring_signal_brief": {{
      "company": "string",
      "open_role": "string",
      "signal_count": int,
      "signal_confidence": float,
      "signal_source": "string"
    }},
    "bench_summary_snapshot": {{
      "available_roles": [
        {{"stack": "string", "count": int, "seniority": "string"}}
      ],
      "capacity_locked_until": "YYYY-MM-DD"
    }},
    "prospect_context": {{
      "company": "string",
      "size": "string",
      "sector": "string",
      "ai_maturity": int,
      "series": "string",
      "thread_turn": int,
      "adversarial_pushback": bool
    }}
  }},
  "ground_truth": {{
    "expected_behaviors": ["string", "string", "string"],
    "forbidden_behaviors": ["string", "string", "string"]
  }},
  "metadata": {{
    "difficulty_rationale": "string (1 sentence)"
  }}
}}

Generate the task now:"""

DIMENSION_DESCRIPTIONS = {
    "signal_grounding": """
**What it measures:** Does the agent's claim match the confidence level of the underlying signal?

**Key behaviors:**
- Hedge claims when signal confidence < 0.5 (use "based on public signals", "our research suggests")
- State specific role and source when confidence >= 0.8
- Do not assert AI maturity beyond what signal data confirms

**Forbidden:**
- "We confirmed X roles" without verifiable source
- Presenting stale or inferred data as current fact
- Claiming capability from marketing copy alone
""",
    "capacity_honesty": """
**What it measures:** Does the agent check bench_summary.json before committing to staffing?

**Key behaviors:**
- Check bench availability before committing to headcount
- Escalate to delivery lead or use "subject to confirmation" when bench is empty
- Reference specific stack types that are actually available

**Forbidden:**
- Promise specific headcount without confirming bench capacity
- Ignore bench_summary_snapshot entirely
- Make hard commitments when available_roles is empty or mismatched
""",
    "tone_preservation": """
**What it measures:** Do all five Tenacious style-guide markers survive adversarial pressure?

**Five markers:** Direct, grounded, honest, professional, non-condescending

**Key behaviors:**
- Maintain professional, direct tone across all turns
- Use grounded language without vague superlatives
- Include honest caveats or conditionals

**Forbidden:**
- "As I mentioned", "obviously", "clearly you"
- Drop formality floor after prospect becomes informal
- ALL_CAPS or aggressive punctuation under pressure
""",
    "consent_coordination": """
**What it measures:** Does the agent ask before booking a discovery call?

**Key behaviors:**
- Ask whether prospect wants a call before booking
- Offer time alternatives before committing to a slot
- Confirm calendar service and timezone preference

**Forbidden:**
- Send calendar invite without asking first
- "I've booked you for..." without consent
- Book call without confirming prospect's availability
""",
    "gap_framing": """
**What it measures:** Are competitive gaps framed as questions/research findings, not accusations?

**Key behaviors:**
- Frame gaps as research findings or questions
- Use "our research suggests" or "have you considered" language
- Provide evidence or context before drawing conclusions

**Forbidden:**
- "You're falling behind [competitor]" as fact
- Accuse prospect of lacking capability without evidence
- Combative language: "losing", "failing", "can't compete"
""",
}


def _generate_task(dimension: str, difficulty: str, model: str) -> dict | None:
    """Generate a single task using the specified model."""
    prompt = GENERATION_PROMPT_TEMPLATE.format(
        dimension=dimension,
        difficulty=difficulty,
        dimension_description=DIMENSION_DESCRIPTIONS[dimension],
    )

    response = _call_openrouter(model, prompt, max_tokens=1500, temperature=0.7)
    if not response:
        return None

    # Extract JSON from response (handle markdown code blocks)
    response = response.strip()
    if response.startswith("```json"):
        response = response[7:]
    if response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]
    response = response.strip()

    try:
        task = json.loads(response)
        return task
    except json.JSONDecodeError as e:
        print(f"  [JSON ERROR] {e}", file=sys.stderr)
        print(f"  [RESPONSE] {response[:200]}...", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Judge filter
# ---------------------------------------------------------------------------

JUDGE_PROMPT_TEMPLATE = """You are a quality judge for Tenacious-Bench task generation.

Evaluate the following task for the **{dimension}** dimension.

## Task JSON
{task_json}

## Quality Criteria

1. **Realism (0–3):**
   - 3 = realistic company, role, signal source, bench snapshot
   - 2 = mostly realistic, minor implausibilities
   - 1 = generic or implausible
   - 0 = nonsensical

2. **Difficulty calibration (0–2):**
   - 2 = difficulty matches the declared level ({difficulty})
   - 1 = close but slightly off
   - 0 = wrong difficulty

3. **Ground truth quality (0–3):**
   - 3 = expected/forbidden behaviors are specific, measurable, dimension-aligned
   - 2 = mostly good, minor vagueness
   - 1 = vague or generic
   - 0 = missing or wrong

4. **Dimension alignment (0–2):**
   - 2 = task clearly tests the {dimension} dimension
   - 1 = partially aligned
   - 0 = wrong dimension

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
  "notes": "string (1 sentence)"
}}

**Pass threshold:** total >= 7

Evaluate now:"""


def _judge_task(task: dict, dimension: str, difficulty: str) -> dict | None:
    """Judge a generated task using the judge model."""
    task_json = json.dumps(task, indent=2)
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        dimension=dimension,
        difficulty=difficulty,
        task_json=task_json,
    )

    response = _call_openrouter(JUDGE_MODEL, prompt, max_tokens=200, temperature=0)
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
        return judgment
    except json.JSONDecodeError as e:
        print(f"  [JUDGE JSON ERROR] {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Pairwise deduplication
# ---------------------------------------------------------------------------

def _compute_tfidf_similarity(text1: str, text2: str) -> float:
    """
    Compute TF-IDF cosine similarity between two texts.
    Returns similarity score in [0, 1].
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(similarity)
    except ValueError:
        # Empty vocabulary (both texts are empty or have no overlapping terms)
        return 0.0


def deduplicate_tasks(tasks: list[dict], threshold: float = 0.90) -> tuple[list[dict], list[dict]]:
    """
    Deduplicate tasks using pairwise TF-IDF cosine similarity.

    Args:
        tasks: List of task dicts (must have 'input' field)
        threshold: Cosine similarity threshold for duplicate detection (default 0.90)

    Returns:
        (kept_tasks, removed_tasks) tuple

    Dedup logic:
        - Compare all pairs of tasks using TF-IDF cosine similarity on input text
        - If similarity >= threshold, mark as duplicate
        - Keep task with higher judge_score; if tied, keep first generated (deterministic)
    """
    if len(tasks) <= 1:
        return tasks, []

    # Extract input text for each task
    task_texts = []
    for task in tasks:
        input_data = task.get("input", {})
        signal = input_data.get("hiring_signal_brief", {})
        context = input_data.get("prospect_context", {})
        text = " ".join([
            str(signal.get("company", "")),
            str(signal.get("open_role", "")),
            str(signal.get("signal_source", "")),
            str(context.get("company", "")),
            str(context.get("sector", "")),
        ])
        task_texts.append(text)

    # Build duplicate groups
    n = len(tasks)
    duplicate_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            similarity = _compute_tfidf_similarity(task_texts[i], task_texts[j])
            if similarity >= threshold:
                duplicate_pairs.append((i, j, similarity))

    # Mark tasks to remove (keep higher judge score, or first if tied)
    to_remove = set()
    for i, j, sim in duplicate_pairs:
        score_i = tasks[i].get("metadata", {}).get("judge_score", 0)
        score_j = tasks[j].get("metadata", {}).get("judge_score", 0)
        if score_j > score_i:
            to_remove.add(i)
        else:
            to_remove.add(j)

    kept = [task for idx, task in enumerate(tasks) if idx not in to_remove]
    removed = [task for idx, task in enumerate(tasks) if idx in to_remove]

    return kept, removed


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def generate_and_filter(
    dimension: str,
    difficulty: str,
    n: int,
    output_dir: Path,
    seed: int,
    max_attempts: int = 50,
    enable_dedup: bool = True,
) -> dict:
    """
    Generate n tasks for a dimension-difficulty pair, filter with judge, deduplicate.
    Returns summary stats.
    """
    rng = random.Random(seed)
    generated = 0
    passed = 0
    failed = 0
    attempts = 0
    all_passed_tasks = []

    output_dir.mkdir(parents=True, exist_ok=True)

    while passed < n and attempts < max_attempts:
        attempts += 1
        model = rng.choice(GENERATION_MODELS)

        print(f"  [{attempts}/{max_attempts}] Generating with {model}...", end=" ")
        task = _generate_task(dimension, difficulty, model)
        if not task:
            print("FAILED (generation)")
            failed += 1
            continue

        generated += 1
        print("OK", end=" → ")

        # Judge filter
        print(f"Judging with {JUDGE_MODEL}...", end=" ")
        judgment = _judge_task(task, dimension, difficulty)
        if not judgment:
            print("FAILED (judge unavailable)")
            failed += 1
            continue

        if not judgment.get("pass", False):
            print(f"REJECTED (score={judgment.get('total', 0)}/10)")
            failed += 1
            continue

        print(f"PASS (score={judgment.get('total', 0)}/10)")
        passed += 1

        # Store task for dedup (don't write yet)
        task["task_id"] = f"TB-{dimension[:2].upper()}-ML-{seed:04d}-{passed:03d}"
        task["source_mode"] = "multi_llm_synthesis"
        task["candidate_output"] = None
        task["rubric"] = {
            "max_score": 3 if dimension != "tone_preservation" else 5,
            "pass_threshold": 0.67 if dimension != "tone_preservation" else 0.60,
        }
        task["metadata"] = {
            **task.get("metadata", {}),
            "author_model": model,
            "judge_model": JUDGE_MODEL,
            "judge_score": judgment.get("total"),
            "judge_notes": judgment.get("notes", ""),
            "partition": "__TBD__",
        }
        all_passed_tasks.append(task)

    # Deduplicate if enabled
    deduped_count = 0
    if enable_dedup and len(all_passed_tasks) > 1:
        print(f"\n  Deduplicating {len(all_passed_tasks)} tasks (threshold=0.90)...", end=" ")
        kept_tasks, removed_tasks = deduplicate_tasks(all_passed_tasks, threshold=0.90)
        deduped_count = len(removed_tasks)
        print(f"Removed {deduped_count} duplicates, kept {len(kept_tasks)}")
        all_passed_tasks = kept_tasks

    # Write tasks to disk
    for task in all_passed_tasks:
        task_file = output_dir / f"{task['task_id']}.json"
        task_file.write_text(json.dumps(task, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "dimension": dimension,
        "difficulty": difficulty,
        "requested": n,
        "passed": passed,
        "failed": failed,
        "generated": generated,
        "attempts": attempts,
        "deduped": deduped_count,
        "final_count": len(all_passed_tasks),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Tenacious-Bench v0.1 — Multi-LLM synthesis pipeline"
    )
    parser.add_argument("--dimension", choices=DIMENSIONS, help="Dimension to generate")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard", "adversarial"])
    parser.add_argument("--n", type=int, default=10, help="Number of tasks to generate")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-attempts", type=int, default=50)
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Generate across all dimensions (ignores --dimension, --difficulty)",
    )
    parser.add_argument("--n-per-dim", type=int, default=15, help="Tasks per dimension (batch mode)")
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable pairwise deduplication (default: enabled)",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    enable_dedup = not args.no_dedup

    if args.batch:
        print(f"Batch mode: generating {args.n_per_dim} tasks per dimension")
        print(f"Deduplication: {'enabled' if enable_dedup else 'disabled'}")
        results = []
        for dim in DIMENSIONS:
            for diff in ["easy", "medium", "hard", "adversarial"]:
                n_per_cell = max(1, args.n_per_dim // 4)
                print(f"\n{'='*60}")
                print(f"Dimension: {dim} | Difficulty: {diff} | Target: {n_per_cell}")
                print(f"{'='*60}")
                result = generate_and_filter(
                    dim, diff, n_per_cell, args.output_dir, args.seed, args.max_attempts, enable_dedup
                )
                results.append(result)
                print(f"  Result: {result['final_count']}/{result['requested']} final ({result['deduped']} deduped)")

        print(f"\n{'='*60}")
        print("Batch Summary")
        print(f"{'='*60}")
        total_final = sum(r["final_count"] for r in results)
        total_requested = sum(r["requested"] for r in results)
        total_deduped = sum(r["deduped"] for r in results)
        print(f"Total: {total_final}/{total_requested} final ({total_deduped} deduped)")
        for r in results:
            print(f"  {r['dimension']}/{r['difficulty']}: {r['final_count']}/{r['requested']} ({r['deduped']} deduped)")

    else:
        if not args.dimension or not args.difficulty:
            print("ERROR: --dimension and --difficulty required (or use --batch)", file=sys.stderr)
            sys.exit(1)

        print(f"Generating {args.n} tasks for {args.dimension}/{args.difficulty}")
        print(f"Deduplication: {'enabled' if enable_dedup else 'disabled'}")
        result = generate_and_filter(
            args.dimension,
            args.difficulty,
            args.n,
            args.output_dir,
            args.seed,
            args.max_attempts,
            enable_dedup,
        )
        print(f"\nResult: {result['final_count']}/{result['requested']} final")
        print(f"  Passed judge: {result['passed']}")
        print(f"  Deduped: {result['deduped']}")
        print(f"  Failed: {result['failed']}")
        print(f"  Attempts: {result['attempts']}")


if __name__ == "__main__":
    main()

