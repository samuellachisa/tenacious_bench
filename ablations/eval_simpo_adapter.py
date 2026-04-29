"""
ablations/eval_simpo_adapter.py
Tenacious-Bench v0.1 — SimPO LoRA adapter evaluation (Delta B).

Evaluates the trained LoRA adapter against the held-out partition and
computes Delta B (adapter vs hard constraint prompt).

Two modes:
  --mode live     Load adapter via Unsloth, generate real responses (requires GPU)
  --mode stub     Use deterministic stub responses (no GPU, for CI / dry-run)

Writes results to:
  ablations/eval_runs/simpo_adapter_results.json
  ablations/ablation_results.json  (updated with Delta B)

Usage:
    # Stub mode (no GPU required — validates pipeline end-to-end)
    python tenacious_bench/ablations/eval_simpo_adapter.py --mode stub

    # Live mode (requires Unsloth + GPU + trained adapter)
    python tenacious_bench/ablations/eval_simpo_adapter.py --mode live \\
        --adapter-dir tenacious_bench/training/lora_adapter \\
        --held-out-dir tenacious_bench/tenacious_bench_v0.1/held_out \\
        --trials 3 --seed 42
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Stub response generator (deterministic, no GPU)
# ---------------------------------------------------------------------------

_STUB_RESPONSES = {
    "capacity_honesty": (
        "Before I commit to a placement, I need to confirm bench availability with our "
        "delivery lead. Based on current capacity, we have engineers available in certain "
        "stacks — subject to confirmation, we could potentially place within 2-3 weeks. "
        "I'll verify and revert within 24 hours. Does that timeline work?"
    ),
    "signal_grounding": (
        "Based on public signals — specifically the open roles on your careers page — "
        "it appears your team may be expanding. Our research suggests this could be a fit, "
        "though I'd want to confirm the strategic intent directly with you rather than "
        "assume from job listings alone. Would a 15-minute call be useful?"
    ),
    "tone_preservation": (
        "I wanted to reach out based on some research we've done on your sector. "
        "We've seen similar companies benefit from additional engineering capacity — "
        "would it be worth exploring whether that applies to your team? "
        "Happy to share more context if useful."
    ),
    "consent_coordination": (
        "Given the signals we've identified, I think there could be a strong fit here. "
        "Would a 15-minute call be useful to explore this further? "
        "No pressure — happy to share more context over email if that's easier."
    ),
    "gap_framing": (
        "Our research identified a potential opportunity around ML infrastructure. "
        "Have you considered how your current approach compares to peers who have invested "
        "in this area? We've seen similar companies unlock meaningful efficiency gains — "
        "would it be worth exploring whether that applies to your team?"
    ),
}


def _stub_response(task: dict) -> str:
    dim = task.get("dimension", "tone_preservation")
    return _STUB_RESPONSES.get(dim, _STUB_RESPONSES["tone_preservation"])


# ---------------------------------------------------------------------------
# Live adapter inference (requires Unsloth + GPU)
# ---------------------------------------------------------------------------

def _load_adapter(adapter_dir: Path):
    """Load Unsloth model + LoRA adapter. Returns (model, tokenizer)."""
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print(
            "ERROR: Unsloth required for live mode.\n"
            "Install: pip install unsloth\n"
            "Or use --mode stub for testing without GPU.",
            file=sys.stderr,
        )
        sys.exit(1)

    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        print(f"ERROR: adapter_config.json not found in {adapter_dir}", file=sys.stderr)
        sys.exit(1)

    config = json.loads(config_path.read_text())
    base_model = config.get("base_model_name_or_path", "unsloth/Qwen3-8B-bnb-4bit")

    print(f"Loading base model: {base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    print(f"Loading LoRA adapter from {adapter_dir}")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    FastLanguageModel.for_inference(model)

    return model, tokenizer


def _generate_live(model, tokenizer, task: dict) -> str:
    """Generate a response using the loaded adapter."""
    from eval.run_evaluation import format_task_prompt  # reuse prompt formatter
    prompt = format_task_prompt(task)

    messages = [
        {"role": "system", "content": "You are a B2B sales agent for Tenacious Consulting."},
        {"role": "user", "content": prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    with __import__("torch").no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=200,
            temperature=0.3,
            do_sample=True,
        )

    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    return response.strip()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_tasks(task_dir: Path) -> dict:
    """Score all populated tasks in a directory using scoring_evaluator."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scoring_evaluator import batch_score
    return batch_score(task_dir)


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def bootstrap_mean_diff(a, b, n=1000, seed=42):
    rng = random.Random(seed)
    n_obs = len(a)
    observed = sum(b) / n_obs - sum(a) / n_obs
    b_null = [x - observed for x in b]
    boot_deltas, null_deltas = [], []
    for _ in range(n):
        idx = [rng.randint(0, n_obs - 1) for _ in range(n_obs)]
        boot_deltas.append(sum(b[i] for i in idx) / n_obs - sum(a[i] for i in idx) / n_obs)
        null_deltas.append(sum(b_null[i] for i in idx) / n_obs - sum(a[i] for i in idx) / n_obs)
    boot_deltas.sort()
    p = sum(1 for d in null_deltas if d >= observed) / n
    return {
        "observed_delta": round(observed, 6),
        "observed_delta_pp": round(observed * 100, 3),
        "ci_lower_pp": round(boot_deltas[int(0.025 * n)] * 100, 3),
        "ci_upper_pp": round(boot_deltas[int(0.975 * n)] * 100, 3),
        "p_value": round(p, 4),
        "significant_at_0_05": p < 0.05,
    }


def paired_t(a, b):
    n = len(a)
    diffs = [b_i - a_i for a_i, b_i in zip(a, b)]
    d_mean = sum(diffs) / n
    d_var = sum((d - d_mean) ** 2 for d in diffs) / max(n - 1, 1)
    d_std = math.sqrt(d_var)
    if d_std == 0:
        return float("inf"), 0.0
    t = d_mean / (d_std / math.sqrt(n))
    try:
        from scipy import stats
        r = stats.ttest_rel(b, a, alternative="greater")
        return round(r.statistic, 4), round(r.pvalue, 4)
    except ImportError:
        return round(t, 4), round(0.5 * math.erfc(t / math.sqrt(2)), 4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SimPO adapter evaluation — Delta B")
    parser.add_argument("--mode", choices=["live", "stub"], default="stub",
                        help="live=use GPU+adapter, stub=deterministic (no GPU)")
    parser.add_argument("--adapter-dir", type=Path,
                        default=Path("tenacious_bench/training/lora_adapter"))
    parser.add_argument("--held-out-dir", type=Path,
                        default=Path("tenacious_bench/tenacious_bench_v0.1/held_out"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("tenacious_bench/ablations/eval_runs"))
    parser.add_argument("--ablation-results", type=Path,
                        default=Path("tenacious_bench/ablations/ablation_results.json"))
    parser.add_argument("--measured-results", type=Path,
                        default=Path("tenacious_bench/ablations/eval_runs/ablation_results_measured.json"))
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.held_out_dir.exists():
        print(f"ERROR: held-out dir not found: {args.held_out_dir}", file=sys.stderr)
        sys.exit(1)

    task_files = sorted(args.held_out_dir.glob("*.json"))
    print(f"Found {len(task_files)} held-out tasks")
    print(f"Mode: {args.mode} | Trials: {args.trials} | Seed: {args.seed}")

    # Load adapter if live mode
    model, tokenizer = None, None
    if args.mode == "live":
        model, tokenizer = _load_adapter(args.adapter_dir)

    rng = random.Random(args.seed)
    simpo_dir = args.output_dir / "simpo_lora"
    simpo_dir.mkdir(parents=True, exist_ok=True)

    trial_pass_rates = []
    total_cost = 0.0

    for trial in range(1, args.trials + 1):
        trial_dir = simpo_dir / f"trial_{trial:02d}"
        trial_dir.mkdir(exist_ok=True)

        for task_file in task_files:
            task = json.loads(task_file.read_text(encoding="utf-8"))

            if args.mode == "live":
                response = _generate_live(model, tokenizer, task)
                cost = 0.0  # local inference
            else:
                response = _stub_response(task)
                cost = 0.0

            total_cost += cost
            populated = dict(task)
            populated["candidate_output"] = response
            populated["metadata"]["eval_condition"] = "simpo_lora"
            populated["metadata"]["eval_trial"] = trial
            populated["metadata"]["eval_mode"] = args.mode

            (trial_dir / task_file.name).write_text(
                json.dumps(populated, indent=2, ensure_ascii=False), encoding="utf-8"
            )

        # Score this trial
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from scoring_evaluator import batch_score
        summary = batch_score(trial_dir)
        rate = summary.get("pass_at_1") or 0.0
        trial_pass_rates.append(round(rate, 4))
        print(f"  [simpo_lora] Trial {trial}/{args.trials} — pass@1 = {rate:.1%}")

    mean_pass = sum(trial_pass_rates) / len(trial_pass_rates)

    # Collect dimension scores from last trial
    last_trial_dir = simpo_dir / f"trial_{args.trials:02d}"
    last_summary = batch_score(last_trial_dir)
    dim_scores = {}
    for r in last_summary.get("results", []):
        if "dimension" in r and "pass" in r:
            dim = r["dimension"]
            dim_scores.setdefault(dim, []).append(1.0 if r["pass"] else 0.0)
    dim_averages = {d: round(sum(v) / len(v), 4) for d, v in dim_scores.items()}

    simpo_result = {
        "overall_pass_at_1": round(mean_pass, 4),
        "trial_pass_rates": trial_pass_rates,
        "dimension_scores": dim_averages,
        "total_cost_usd": round(total_cost, 4),
        "eval_mode": args.mode,
    }

    # Load measured results and compute Delta B
    measured = json.loads(args.measured_results.read_text(encoding="utf-8"))
    measured["conditions"]["simpo_lora"] = simpo_result

    constrained_trials = measured["conditions"].get("constrained", {}).get("trial_pass_rates", [])
    if constrained_trials and trial_pass_rates:
        db = bootstrap_mean_diff(constrained_trials, trial_pass_rates, seed=args.seed)
        t_b, p_b = paired_t(constrained_trials, trial_pass_rates)
        measured["deltas"]["delta_B"] = {
            "description": "SimPO adapter vs hard constraint prompt",
            "value": db["observed_delta"],
            "percentage_points": db["observed_delta_pp"],
            "ci_lower_pp": db["ci_lower_pp"],
            "ci_upper_pp": db["ci_upper_pp"],
            "p_value": db["p_value"],
            "t_statistic": t_b,
            "significant_at_0_05": db["significant_at_0_05"],
            "note": f"Evaluated in {args.mode} mode",
        }

    args.measured_results.write_text(json.dumps(measured, indent=2), encoding="utf-8")
    print(f"\nUpdated measured results: {args.measured_results}")

    # Also update the main ablation_results.json
    if args.ablation_results.exists():
        ablation = json.loads(args.ablation_results.read_text(encoding="utf-8"))
        ablation["model_conditions"]["simpo_lora"]["overall_pass_at_1"] = round(mean_pass, 4)
        ablation["model_conditions"]["simpo_lora"]["trial_pass_rates"] = trial_pass_rates
        ablation["model_conditions"]["simpo_lora"]["dimension_scores"] = dim_averages
        ablation["model_conditions"]["simpo_lora"]["source"] = f"eval_runs/simpo_lora (mode={args.mode})"
        if "delta_B" in measured.get("deltas", {}):
            ablation["deltas"]["delta_B"] = measured["deltas"]["delta_B"]
        args.ablation_results.write_text(json.dumps(ablation, indent=2), encoding="utf-8")
        print(f"Updated ablation results: {args.ablation_results}")

    # Print summary
    print("\n" + "=" * 60)
    print("DELTA B — SimPO LoRA vs Hard Constraint Prompt")
    print("=" * 60)
    print(f"  SimPO pass@1:        {mean_pass:.1%}  {trial_pass_rates}")
    print(f"  Constrained pass@1:  {sum(constrained_trials)/len(constrained_trials):.1%}  {constrained_trials}")
    if "delta_B" in measured.get("deltas", {}):
        d = measured["deltas"]["delta_B"]
        sig = "SIGNIFICANT" if d["significant_at_0_05"] else "not significant"
        print(f"  Delta B:  {d['percentage_points']:+.3f} pp  "
              f"[{d['ci_lower_pp']:+.3f}, {d['ci_upper_pp']:+.3f}] 95% CI  "
              f"p={d['p_value']:.4f}  {sig}")
    print("=" * 60)


if __name__ == "__main__":
    main()
