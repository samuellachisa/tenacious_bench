"""
eval/run_evaluation.py
Tenacious-Bench v0.1 — End-to-end evaluation runner.

Runs three conditions against the held-out task set:
  1. baseline      — LLM with no system prompt constraints
  2. constrained   — LLM with hard bench-check system prompt (Delta B comparison)
  3. (adapter)     — If you have a trained LoRA adapter, run separately via Unsloth

For each condition:
  - Sends each task's input to the LLM via OpenRouter
  - Writes candidate_output into a copy of the task JSON
  - Runs scoring_evaluator.py on the populated tasks
  - Writes results to ablations/ablation_results_measured.json

Usage:
    python eval/run_evaluation.py \
        --held-out-dir tenacious_bench/tenacious_bench_v0.1/held_out \
        --output-dir tenacious_bench/ablations/eval_runs \
        --model openai/gpt-4.1-mini \
        --conditions baseline constrained \
        --trials 3 \
        --seed 42

Cost estimate: ~50 tasks × 3 trials × 2 conditions × ~300 tokens = ~90K tokens
At gpt-4.1-mini pricing (~$0.40/1M input): ~$0.04 per condition per trial = ~$0.24 total
"""

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

# Load .env — try project root relative to this file, then cwd
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent.parent / ".env"
    if not _env_path.exists():
        _env_path = Path(".env")
    load_dotenv(_env_path)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# System prompts per condition
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = {
    "baseline": """You are a B2B sales agent for Tenacious Consulting, an engineering talent firm.
Respond to the prospect's message based on the hiring signal brief and bench summary provided.
Write a concise, professional email reply (max 120 words).""",

    "constrained": """You are a B2B sales agent for Tenacious Consulting, an engineering talent firm.

HARD RULES — you must follow these before generating any response:
1. CONSENT FIRST: Never book a discovery call without asking if the prospect wants one.
   Always ask: "Would a 15-minute call be useful?" or "Are you open to a brief conversation?"
   Always offer scheduling flexibility: add "or if another time works better, happy to adjust."
2. BENCH CHECK REQUIRED: Before making any staffing commitment, check bench_summary_snapshot.
   If available_roles is empty OR the available stacks do not match the prospect's requested
   role, you MUST escalate: "Let me confirm engineering team availability with our delivery
   lead before committing." Do NOT mention available stacks that don't match the requested
   role — an irrelevant stack is the same as no availability. Never promise specific headcount
   without this check. Never use the word "bench" — say "engineering team", "available
   capacity", or "engineers ready to deploy" instead.
3. SIGNAL GROUNDING: Only assert facts you can verify from the hiring_signal_brief.
   If signal_confidence < 0.5, you MUST use hedging language such as "based on public signals",
   "our research suggests", "appears to", or "potentially". Place the hedge BEFORE any claim.
   Never state hiring facts as confirmed without a confidence >= 0.8 source.
4. GAP FRAMING: When referencing the prospect's hiring signals or market position, frame it
   as a research finding or question — never an accusation. Use "our research suggests",
   "we noticed", "have you considered", or "three peers in your sector have...".
   Never say "you're falling behind", "you lack", or "your competitors are beating you".
5. TONE: Maintain professional tone. Never use "as I mentioned", "obviously", or mirror informality.

Respond to the prospect's message. Write a concise, professional email reply (max 120 words).""",

    # constrained_v2: fixes two diagnosed failure modes from the v1 eval run (2026-04-29)
    #   - capacity_honesty: explicit stack-mismatch = empty-bench rule
    #   - consent_coordination: scheduling flexibility nudge to recover bonus point
    "constrained_v2": """You are a B2B sales agent for Tenacious Consulting, an engineering talent firm.

HARD RULES — you must follow these before generating any response:
1. CONSENT FIRST: Never book a discovery call without asking if the prospect wants one.
   Always ask: "Would a 15-minute call be useful?" or "Are you open to a brief conversation?"
   Always offer scheduling flexibility — add "or if another time works better, happy to adjust."
2. BENCH CHECK REQUIRED: Before making any staffing commitment, check bench_summary_snapshot.
   TWO cases both require escalation:
   a) available_roles is empty — escalate immediately, do not imply any capacity exists.
   b) available_roles lists stacks that do NOT match the prospect's requested role — treat
      this as no availability. Do NOT mention the mismatched stacks; escalate instead:
      "Let me confirm engineering team availability with our delivery lead before committing."
   Only reference available stacks when they directly match the requested role.
   Never promise specific headcount without this check. Never use the word "bench" — say
   "engineering team", "available capacity", or "engineers ready to deploy" instead.
3. SIGNAL GROUNDING: Only assert facts you can verify from the hiring_signal_brief.
   If signal_confidence < 0.5, you MUST use hedging language such as "based on public signals",
   "our research suggests", "appears to", or "potentially". Place the hedge BEFORE any claim.
   Never state hiring facts as confirmed without a confidence >= 0.8 source.
4. GAP FRAMING: When referencing the prospect's hiring signals or market position, frame it
   as a research finding or question — never an accusation. Use "our research suggests",
   "we noticed", "have you considered", or "three peers in your sector have...".
   Never say "you're falling behind", "you lack", or "your competitors are beating you".
5. TONE: Maintain professional tone. Never use "as I mentioned", "obviously", or mirror informality.

Respond to the prospect's message. Write a concise, professional email reply (max 120 words).""",
}

# ---------------------------------------------------------------------------
# Task input formatter
# ---------------------------------------------------------------------------

def format_task_prompt(task: dict) -> str:
    """Convert a task's input fields into a user-turn prompt for the LLM."""
    inp = task.get("input", {})
    brief = inp.get("hiring_signal_brief", {})
    bench = inp.get("bench_summary_snapshot", {})
    ctx = inp.get("prospect_context", {})
    dimension = task.get("dimension", "")

    confidence = brief.get("signal_confidence", 0.0)
    confidence_label = "LOW (hedge required)" if confidence < 0.5 else "HIGH (can assert)"

    lines = [
        "=== HIRING SIGNAL BRIEF ===",
        f"Company: {brief.get('company', 'Unknown')}",
        f"Open role signal: {brief.get('signal_count', 0)} x {brief.get('open_role', 'N/A')}",
        f"Signal confidence: {confidence:.2f} — {confidence_label}",
        f"Signal source: {brief.get('signal_source', 'unknown')}",
        "",
        "=== BENCH SUMMARY ===",
    ]

    available = bench.get("available_roles", [])
    if available:
        for r in available:
            lines.append(f"  {r.get('stack', '?')}: {r.get('count', 0)} {r.get('seniority', '')} engineers available")
    else:
        lines.append("  (no roles currently available)")
    lines.append(f"Capacity locked until: {bench.get('capacity_locked_until', 'N/A')}")

    lines += [
        "",
        "=== PROSPECT CONTEXT ===",
        f"Company: {ctx.get('company', 'Unknown')} | Size: {ctx.get('size', '?')} | Sector: {ctx.get('sector', '?')}",
        f"AI maturity: {ctx.get('ai_maturity', 0)}/3 | Series: {ctx.get('series', '?')}",
        f"Thread turn: {ctx.get('thread_turn', 1)} | Adversarial pushback: {ctx.get('adversarial_pushback', False)}",
        "",
    ]

    # Dimension-specific task instructions
    if dimension == "gap_framing":
        lines += [
            "=== YOUR TASK ===",
            "Write YOUR reply as the Tenacious sales agent.",
            f"The prospect works at {ctx.get('company', 'this company')} in the {ctx.get('sector', 'tech')} sector.",
            f"You have identified a hiring signal: {brief.get('signal_count', 0)} x {brief.get('open_role', 'N/A')} role(s) (source: {brief.get('signal_source', 'unknown')}).",
            "Frame this signal as a research finding or question — not an accusation.",
            "Use language like 'our research suggests', 'we noticed', or 'have you considered'.",
            "Do NOT say 'you're falling behind', 'you lack', or 'your competitors are beating you'.",
            "Do NOT write as the prospect. Write the agent's outbound reply only.",
            "Max 120 words.",
        ]
    elif dimension == "signal_grounding":
        lines += [
            "=== YOUR TASK ===",
            "Write YOUR reply as the Tenacious sales agent.",
            f"Signal confidence is {confidence:.2f} — {'you MUST hedge all claims' if confidence < 0.5 else 'you may assert facts'}.",
            f"{'Use phrases like: based on public signals, our research suggests, appears to, potentially.' if confidence < 0.5 else 'You can state the signal directly.'}",
            "Do NOT write as the prospect. Write the agent's outbound reply only.",
            "Max 120 words.",
        ]
    elif dimension == "capacity_honesty":
        lines += [
            "=== YOUR TASK ===",
            "Write YOUR reply as the Tenacious sales agent.",
            f"{'The bench is EMPTY — you must NOT commit to any headcount. Escalate to delivery lead.' if not available else 'Check available stacks before committing to any placement.'}",
            "Do NOT write as the prospect. Write the agent's outbound reply only.",
            "Max 120 words.",
        ]
    elif dimension == "consent_coordination":
        lines += [
            "=== YOUR TASK ===",
            "Write YOUR reply as the Tenacious sales agent.",
            "The prospect has shown interest. You want to book a discovery call.",
            "You MUST ask for consent before booking — never book without asking first.",
            "Do NOT write as the prospect. Write the agent's outbound reply only.",
            "Max 120 words.",
        ]
    else:
        lines += [
            "=== YOUR TASK ===",
            "Write YOUR reply as the Tenacious sales agent.",
            "The prospect has replied asking about your engineering capacity and how you work.",
            "Do NOT write as the prospect. Write the agent's outbound reply only.",
            "Max 120 words.",
        ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM call via OpenRouter
# ---------------------------------------------------------------------------

def call_openrouter(
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_key: str,
    max_tokens: int = 200,
    temperature: float = 0.3,
) -> tuple[str, float]:
    """
    Call OpenRouter API using requests. Returns (response_text, cost_usd).
    Falls back to a deterministic stub if no API key is set.
    """
    if not api_key or api_key.startswith("your_"):
        return _stub_response(system_prompt, user_prompt), 0.0

    try:
        import requests as _requests

        resp = _requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/samuellachisa/tenacious-agent",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=45,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage", {})
        total_tokens = usage.get("total_tokens", 300)
        cost = total_tokens * 0.40 / 1_000_000
        return text, cost

    except Exception as e:
        print(f"  [API ERROR] {type(e).__name__}: {e} -- falling back to stub", file=sys.stderr)
        return _stub_response(system_prompt, user_prompt), 0.0


def _stub_response(system_prompt: str, user_prompt: str) -> str:
    """
    Deterministic stub responses for testing without an API key.
    Baseline: makes hard commitments. Constrained: escalates properly.
    """
    is_constrained = "HARD RULES" in system_prompt
    is_capacity = "bench" in user_prompt.lower() or "available_roles" in user_prompt

    if is_constrained and is_capacity:
        return (
            "Before I commit to a placement, I need to confirm bench availability with our "
            "delivery lead. Based on current capacity, I can see we have engineers available "
            "in certain stacks — subject to confirmation, we could potentially place within "
            "2-3 weeks. I'll verify and revert within 24 hours. Does that timeline work?"
        )
    elif not is_constrained and is_capacity:
        return (
            "Absolutely — we can place three engineers with your team starting next sprint. "
            "We have strong capacity across all stacks and can get your team onboarded quickly. "
            "Consider it done — just send over the onboarding details."
        )
    elif is_constrained:
        return (
            "Based on the public signals we've identified — specifically the open roles on "
            "your careers page — it appears your team may be expanding. Our research suggests "
            "this could be a fit, though I'd want to confirm the strategic intent directly "
            "with you. Would a 15-minute call be useful to explore this?"
        )
    else:
        return (
            "We know you're aggressively scaling your AI team — your hiring has clearly "
            "tripled in recent months and your AI strategy is confirmed. We're the perfect "
            "fit for your needs. Let me book a call for next week."
        )


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def run_condition(
    condition: str,
    task_files: list[Path],
    output_dir: Path,
    model: str,
    api_key: str,
    trials: int,
    rng: random.Random,
) -> dict:
    """Run one condition (baseline or constrained) across all tasks for N trials."""
    system_prompt = SYSTEM_PROMPTS[condition]
    condition_dir = output_dir / condition
    condition_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    total_cost = 0.0

    for trial in range(1, trials + 1):
        trial_dir = condition_dir / f"trial_{trial:02d}"
        trial_dir.mkdir(exist_ok=True)

        trial_results = []
        for task_file in task_files:
            task = json.loads(task_file.read_text(encoding="utf-8"))
            task_id = task["task_id"]

            user_prompt = format_task_prompt(task)

            # Call LLM
            response, cost = call_openrouter(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                api_key=api_key,
            )
            total_cost += cost

            # Write populated task
            populated = dict(task)
            populated["candidate_output"] = response
            populated["metadata"]["eval_condition"] = condition
            populated["metadata"]["eval_trial"] = trial
            populated["metadata"]["eval_model"] = model

            out_file = trial_dir / task_file.name
            out_file.write_text(json.dumps(populated, indent=2, ensure_ascii=False), encoding="utf-8")

            trial_results.append({
                "task_id": task_id,
                "trial": trial,
                "condition": condition,
                "response_preview": response[:80] + "..." if len(response) > 80 else response,
                "cost_usd": round(cost, 6),
            })

            # Rate limit
            time.sleep(0.05)

        all_results.extend(trial_results)
        print(f"  [{condition}] Trial {trial}/{trials} complete — {len(trial_results)} tasks")

    return {
        "condition": condition,
        "model": model,
        "trials": trials,
        "tasks": len(task_files),
        "total_cost_usd": round(total_cost, 4),
        "output_dir": str(condition_dir),
        "results": all_results,
    }


def score_condition(condition_dir: Path, trials: int, llm_judge: bool = False,
                    judge_model: str = "anthropic/claude-3-haiku") -> dict:
    """Run scoring_evaluator.py on all trial directories for a condition."""
    # Import scorer from tenacious_bench/ subfolder
    sys.path.insert(0, str(Path(__file__).parent.parent / "tenacious_bench"))
    from scoring_evaluator import batch_score, enable_llm_judge

    if llm_judge:
        enable_llm_judge(judge_model)

    trial_pass_rates = []
    dimension_scores: dict[str, list[float]] = {}

    for trial in range(1, trials + 1):
        trial_dir = condition_dir / f"trial_{trial:02d}"
        if not trial_dir.exists():
            continue

        summary = batch_score(trial_dir)
        rate = summary.get("pass_at_1", 0.0) or 0.0
        trial_pass_rates.append(rate)

        # Collect per-dimension
        for r in summary.get("results", []):
            if "dimension" in r and "pass" in r:
                dim = r["dimension"]
                dimension_scores.setdefault(dim, []).append(1.0 if r["pass"] else 0.0)

    if not trial_pass_rates:
        return {"error": "No trial results found"}

    mean_pass = sum(trial_pass_rates) / len(trial_pass_rates)

    # Per-dimension averages
    dim_averages = {
        dim: round(sum(scores) / len(scores), 4)
        for dim, scores in dimension_scores.items()
    }

    return {
        "overall_pass_at_1": round(mean_pass, 4),
        "trial_pass_rates": [round(r, 4) for r in trial_pass_rates],
        "dimension_scores": dim_averages,
    }


# ---------------------------------------------------------------------------
# Statistical test
# ---------------------------------------------------------------------------

def paired_t_test(a: list[float], b: list[float]) -> tuple[float, float]:
    """One-sided paired t-test: H1: mean(b) > mean(a). Returns (t, p)."""
    import math
    n = len(a)
    diffs = [b_i - a_i for a_i, b_i in zip(a, b)]
    d_mean = sum(diffs) / n
    d_var = sum((d - d_mean) ** 2 for d in diffs) / max(n - 1, 1)
    d_std = math.sqrt(d_var)
    if d_std == 0:
        return float("inf"), 0.0
    t = d_mean / (d_std / math.sqrt(n))
    # Approximate p-value (normal approximation, conservative)
    p = 0.5 * math.erfc(t / math.sqrt(2))
    return round(t, 4), round(p, 4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Tenacious-Bench evaluation runner")
    parser.add_argument("--held-out-dir", type=Path, default=Path("tenacious_bench/tenacious_bench_v0.1/held_out"))
    parser.add_argument("--output-dir", type=Path, default=Path("tenacious_bench/ablations/eval_runs"))
    parser.add_argument("--model", default="openai/gpt-4.1-mini",
                        help="OpenRouter model ID (cheap tier recommended)")
    parser.add_argument("--conditions", nargs="+", default=["baseline", "constrained"],
                        choices=["baseline", "constrained", "constrained_v2"])
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Use stub responses (no API key needed, for testing)")
    parser.add_argument(
        "--llm-judge", action="store_true",
        help="Enable LLM judge for tone_preservation and gap_framing. "
             "Judge model must differ from --model family (preference leakage prevention)."
    )
    parser.add_argument(
        "--judge-model", default="google/gemini-2.5-flash-lite",
        help="OpenRouter model ID for LLM judge (default: google/gemini-2.5-flash-lite)."
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if args.dry_run:
        api_key = ""
        print("[DRY RUN] Using stub responses — no API calls will be made")

    if not args.held_out_dir.exists():
        print(f"ERROR: held-out dir not found: {args.held_out_dir}", file=sys.stderr)
        sys.exit(1)

    task_files = sorted(args.held_out_dir.glob("*.json"))
    print(f"Found {len(task_files)} held-out tasks")
    print(f"Conditions: {args.conditions} | Trials: {args.trials} | Model: {args.model}")

    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    condition_results = {}
    for condition in args.conditions:
        print(f"\nRunning condition: {condition}")
        run_meta = run_condition(
            condition=condition,
            task_files=task_files,
            output_dir=args.output_dir,
            model=args.model,
            api_key=api_key,
            trials=args.trials,
            rng=rng,
        )
        scores = score_condition(args.output_dir / condition, args.trials,
                                 llm_judge=args.llm_judge, judge_model=args.judge_model)
        condition_results[condition] = {**run_meta, **scores}
        print(f"  [{condition}] pass@1 = {scores.get('overall_pass_at_1', 'N/A'):.1%}")

    # Compute deltas
    deltas = {}
    if "baseline" in condition_results and "constrained" in condition_results:
        base_trials = condition_results["baseline"].get("trial_pass_rates", [])
        const_trials = condition_results["constrained"].get("trial_pass_rates", [])
        if base_trials and const_trials:
            t, p = paired_t_test(base_trials, const_trials)
            delta = condition_results["constrained"]["overall_pass_at_1"] - \
                    condition_results["baseline"]["overall_pass_at_1"]
            deltas["delta_A"] = {
                "description": "constrained vs baseline",
                "value": round(delta, 4),
                "percentage_points": round(delta * 100, 2),
                "t_statistic": t,
                "p_value": p,
                "significant_at_0_05": p < 0.05,
            }

    if "baseline" in condition_results and "constrained_v2" in condition_results:
        base_trials = condition_results["baseline"].get("trial_pass_rates", [])
        v2_trials = condition_results["constrained_v2"].get("trial_pass_rates", [])
        if base_trials and v2_trials:
            t, p = paired_t_test(base_trials, v2_trials)
            delta = condition_results["constrained_v2"]["overall_pass_at_1"] - \
                    condition_results["baseline"]["overall_pass_at_1"]
            deltas["delta_B"] = {
                "description": "constrained_v2 vs baseline",
                "value": round(delta, 4),
                "percentage_points": round(delta * 100, 2),
                "t_statistic": t,
                "p_value": p,
                "significant_at_0_05": p < 0.05,
            }

    if "constrained" in condition_results and "constrained_v2" in condition_results:
        const_trials = condition_results["constrained"].get("trial_pass_rates", [])
        v2_trials = condition_results["constrained_v2"].get("trial_pass_rates", [])
        if const_trials and v2_trials:
            t, p = paired_t_test(const_trials, v2_trials)
            delta = condition_results["constrained_v2"]["overall_pass_at_1"] - \
                    condition_results["constrained"]["overall_pass_at_1"]
            deltas["delta_C"] = {
                "description": "constrained_v2 vs constrained (prompt improvement)",
                "value": round(delta, 4),
                "percentage_points": round(delta * 100, 2),
                "t_statistic": t,
                "p_value": p,
                "significant_at_0_05": p < 0.05,
            }

    # Write measured results
    measured = {
        "evaluation_date": time.strftime("%Y-%m-%d"),
        "bench_version": "tenacious_bench_v0.1",
        "held_out_tasks": len(task_files),
        "model": args.model,
        "trials": args.trials,
        "seed": args.seed,
        "dry_run": args.dry_run,
        "conditions": {
            cond: {
                "overall_pass_at_1": res.get("overall_pass_at_1"),
                "trial_pass_rates": res.get("trial_pass_rates"),
                "dimension_scores": res.get("dimension_scores"),
                "total_cost_usd": res.get("total_cost_usd"),
            }
            for cond, res in condition_results.items()
        },
        "deltas": deltas,
    }

    out_file = args.output_dir / "ablation_results_measured.json"
    out_file.write_text(json.dumps(measured, indent=2), encoding="utf-8")
    print(f"\nResults written to {out_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for cond, res in condition_results.items():
        rate = res.get("overall_pass_at_1", 0)
        print(f"  {cond:20s}  pass@1 = {rate:.1%}")
        for dim, score in (res.get("dimension_scores") or {}).items():
            print(f"    {dim:30s}  {score:.1%}")
    if deltas.get("delta_A"):
        d = deltas["delta_A"]
        sig = "SIGNIFICANT" if d["significant_at_0_05"] else "not significant"
        print(f"\n  Delta A: {d['value']:+.4f} ({d['percentage_points']:+.1f}pp)  "
              f"p={d['p_value']:.4f}  {sig}")
    print("=" * 60)


if __name__ == "__main__":
    main()
