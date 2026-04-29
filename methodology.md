# Methodology — Tenacious-Bench v0.1 and SimPO Training
**Week 11 | Acts I–IV**

---

## Path Declaration

**Path B — Preference-tuned judge / critic (SimPO)**

Declared on Day 1. Justification: the `bench_over_commitment` failure mode (trigger rate 0.45, expected loss $821/100 leads) is a *preference ordering* defect, not a knowledge gap. The agent knows the bench check exists but prefers the fluent hard commitment over the cautious escalation. SFT teaches the surface form; SimPO penalizes the failure pattern at the token level. See `methodology_rationale.md` for full justification with trace citations.

---

## Dataset Authoring

### Four Authoring Modes

| Mode | Share | Count | Description |
|------|-------|-------|-------------|
| trace_derived | ~30% | 75 | Derived from Week 10 `eval/trace_log.jsonl` patterns |
| programmatic | ~30% | 75 | Parametric sweep across fixture pools (seed=42) |
| multi_llm_synthesis | ~25% | 62 | Generated via OpenRouter cheap-tier models |
| hand_authored | ~15% | 38 | Written by primary author to defeat Week 10 agent |

### Partitioning Protocol

- **Train:** 50% (125 tasks) — used for SimPO pair generation
- **Dev:** 30% (75 tasks) — used for ablations and rubric calibration
- **Held-out:** 20% (50 tasks) — sealed; used only for final evaluation

Partitioning was performed by `generation_scripts/generate_dataset.py` with `--seed 42`. The held-out partition was sealed before any training run began.

### Contamination Check Results

| Check | Threshold | Result |
|-------|-----------|--------|
| N-gram overlap (8-gram) | < 1.0 (no exact match) | CLEAN |
| Cosine similarity (TF-IDF) | < 0.85 | CLEAN |
| Violations | 0 | CLEAN |

Run: `python contamination_check.py --bench-dir tenacious_bench_v0.1 --reference-file eval/trace_log.jsonl`

### LLM-as-Judge Rotation Policy

To prevent preference leakage (Li et al., 2025):
- **Generation:** DeepSeek V3 / Qwen3-Next (cheap tier via OpenRouter)
- **Quality filtering:** DeepSeek Chat (different family from generation model)
- **Spot-check calibration (10% sample):** Claude Sonnet / GPT-4.1 class
- **Held-out evaluation:** Claude Sonnet / GPT-4.1 class

The same model family was never used to both generate and judge the same task.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Objective | SimPO (CPOTrainer, loss_type="simpo") |
| Base model | Qwen3-8B-Instruct (unsloth/Qwen3-8B-bnb-4bit) |
| LoRA rank | 16 (alpha=32) |
| Target modules | q_proj, v_proj, k_proj, o_proj |
| Epochs | 3 |
| Learning rate | 5e-6 (cosine decay, warmup_ratio=0.05) |
| Batch size | 4 (grad_accum=8, effective=32) |
| SimPO β | 2.0 |
| SimPO γ | 0.5 |
| Training pairs | 200 |
| Eval split | 20% of pairs (40 pairs) |
| Seed | 42 |
| Framework | Unsloth + TRL CPOTrainer |
| Hardware | Google Colab T4 (free tier) |
| Estimated GPU-hours | ~1.8h |
| Cost | $0.00 |

---

## Evaluation Protocol

All evaluation runs used `scoring_evaluator.py` against `tenacious_bench_v0.1/held_out/` (50 tasks).

Three conditions evaluated:
1. **Baseline:** Qwen3-8B-Instruct, no adapter, no constraint prompt
2. **Hard constraint prompt:** Qwen3-8B-Instruct + system prompt instructing bench check
3. **SimPO LoRA:** Qwen3-8B-Instruct + trained adapter

Statistical test: one-sided paired t-test on 50 tasks × 3 trials. Significance threshold: p < 0.05.

---

## Cost Log

| Date | Bucket | Model | Purpose | Cost |
|------|--------|-------|---------|------|
| 2026-04-29 | Dataset authoring | DeepSeek Chat (OpenRouter) | Judge filtering 250 tasks | ~$0.80 |
| 2026-04-29 | Dataset authoring | Qwen3-Next (OpenRouter) | Bulk task generation | ~$1.20 |
| 2026-04-29 | Training | Colab T4 | SimPO LoRA training | $0.00 |
| 2026-04-29 | Held-out eval | Claude Sonnet (OpenRouter) | Sealed slice scoring (3 passes) | ~$2.50 |
| **Total** | | | | **~$4.50** |

Budget envelope: $10.00. Remaining: ~$5.50.

No τ²-Bench retail re-runs. Week 10 score (72.67%) reused as informational reference per spec.
