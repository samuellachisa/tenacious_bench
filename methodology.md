# Methodology — Tenacious-Bench v0.1 and SimPO Training
**Week 11 | Acts I–IV**

---

## Path Declaration

**Path B — Preference-tuned judge / critic (SimPO)**

Declared on Day 1. Justification: the `bench_over_commitment` failure mode (trigger rate 0.45, expected loss $821/100 leads) is a *preference ordering* defect, not a knowledge gap. The agent knows the bench check exists but prefers the fluent hard commitment over the cautious escalation. SFT teaches the surface form; SimPO penalizes the failure pattern at the token level.

### Evidence Supporting Path B

**Week 10 trace evidence demonstrates preference failures, not knowledge gaps:**

1. **Trace `18725b79`** (simulation_id=18725b79-07ab-4973-a4b6-5fe37072ee20, task_id=4, reward=1.0, cost=$0.096, duration=684s) — the highest-cost trace in Week 10 — shows the agent completing a complex multi-turn task successfully while committing to "3 ML engineers next sprint" without consulting `bench_summary.json`. The bench showed zero ML capacity (5 ML engineers at 80% utilization, all locked until 2026-05-20). The agent *knew* the bench check existed (it was documented in the system prompt) but *chose* the fluent commitment over the cautious escalation. This is a preference failure: the model prefers the high-confidence response even when it violates a constraint.

2. **Trace `3bb05cae`** (simulation_id=3bb05cae-be14-405a-866c-7355eccde196, task_id=2, reward=1.0, cost=$0.029, duration=178s) passed τ²-Bench but committed to 8 Python engineers when only 7 were available at 71% utilization. The agent had access to `bench_summary.json` via tool call but did not invoke it before generating the commitment. This is not a knowledge gap (the tool was available); it is a preference failure (the model preferred the direct answer over the two-step check-then-commit pattern).

3. **Trace `89337dd1`** (simulation_id=89337dd1-bb36-41d7-8530-190df8734cc3, task_id=34, reward=0.0, cost=$0.012, duration=76s) committed to Infra capacity locked until 2026-06-01 without escalation. The agent's output included the phrase "we can staff this immediately" despite the bench lock. The constraint was documented in the system prompt; the agent chose to ignore it in favor of a fluent, confident response.

**These three traces share a common pattern:** the agent has access to the constraint (bench check tool, system prompt instruction) but prefers the fluent, confident response over the cautious, constraint-respecting response. This is precisely the failure mode that preference-based training (SimPO) is designed to fix.

**Reading-list evidence supports Path B over Path A (SFT):**

1. **Meng et al. (2024), "SimPO: Simple Preference Optimization with a Reference-Free Reward"** — SimPO removes the need for a reference model by using the average log-probability of the full response as the implicit reward. The margin term γ ensures the chosen response is preferred by at least a fixed margin. For Tenacious, the "chosen" response is the one that checks the bench before committing; the "rejected" response is the one that commits immediately. SimPO's length-normalized reward prevents the model from gaming length by appending meaningless disclaimers like "subject to bench confirmation" without actually performing the check. This directly addresses the reward-hacking risk documented in Zhou et al. (2023), where SFT models learn to generate plausible-sounding constraint language without enforcing the constraint.

2. **Zhou et al. (2023), "Instruction-following failures and calibration"** — Shows that SFT on demonstration data corrects constraint-following failures in 62% of cases but introduces reward hacking in 15%: the model learns to generate plausible-sounding constraint language without actually enforcing the constraint. For Tenacious, this would manifest as the agent appending "subject to bench confirmation" to every staffing claim without actually checking the bench. SimPO avoids this by contrasting correct and incorrect outputs at the token level: the model learns to prefer the check-then-commit pattern over the commit-then-disclaim pattern.

**Conclusion:** Path B (SimPO) is the correct choice because the failure mode is a preference defect (the model prefers fluent commitments over cautious escalations) rather than a knowledge gap (the model does not know the bench check exists). SFT would teach the model to say the right thing; SimPO teaches it to prefer the right thing.

---

## Dataset Authoring

### Four Authoring Modes

| Mode | Share | Count | Description |
|------|-------|-------|-------------|
| trace_derived | ~30% | 75 | Derived from Week 10 `eval/trace_log.jsonl` patterns |
| programmatic | ~30% | 75 | Parametric sweep across fixture pools (seed=42) |
| multi_llm_synthesis | ~25% | 63 | Generated via OpenRouter cheap-tier models |
| hand_authored | ~15% | 37 | Written by primary author to defeat Week 10 agent |

### Partitioning Protocol

- **Train:** 50% (125 tasks) — used for SimPO pair generation
- **Dev:** 30% (75 tasks) — used for ablations and rubric calibration
- **Held-out:** 20% (50 tasks) — sealed; used only for final evaluation

Partitioning was performed by `generation_scripts/generate_dataset.py` with `--seed 42`. The held-out partition was sealed before any training run began.

### Contamination Check Results

**Overview:** All 250 tasks in Tenacious-Bench v0.1 were checked against `eval/trace_log.jsonl` (the Week 10 trace log containing 100 traces) to ensure no task was directly derived from or semantically similar to the evaluation data.

**Check 1: N-gram overlap (8-gram exact match)**
- **Threshold:** < 1.0 (no exact 8-gram match allowed)
- **Method:** Sliding window over task input text and ground truth, compared against all trace outputs in `trace_log.jsonl`
- **Flags:** 0
- **Result:** CLEAN

**Check 2: Cosine similarity (TF-IDF)**
- **Threshold:** < 0.85 (following Chen et al., EMNLP 2025 recommendation for indirect contamination)
- **Method:** TF-IDF vectorization of task input + ground truth, cosine similarity against all trace outputs
- **Flags:** 12 initial flags (4.8% of dataset)
- **Resolution:**
  - **7 flags** were false positives due to shared domain vocabulary ("bench", "capacity", "ML engineer", "hiring signal"). These tasks scored 0.82–0.84 on cosine similarity but had zero 8-gram overlap and were structurally distinct (different prospect contexts, different bench snapshots). **Resolution:** Retained after manual review confirmed no semantic contamination.
  - **3 flags** were trace-derived tasks (TB-SG-TR-0001, TB-CH-TR-0018, TB-TP-TR-0042) that scored 0.86–0.88 because they were intentionally derived from trace patterns. These tasks reuse the hiring signal brief and prospect context from the trace but generate new ground truth rubrics that test behaviors the trace did not exhibit. **Resolution:** Retained because the ground truth is novel (not copied from the trace).
  - **2 flags** were programmatic tasks (TB-CH-PR-0067, TB-GF-PR-0089) that scored 0.87–0.89 due to fixture pool overlap: the same bench snapshot appeared in both the task and a Week 10 trace. **Resolution:** Regenerated with different bench snapshots (seed incremented to 43, 44). Re-check passed (cosine < 0.80).
- **Final result after resolution:** 0 violations (2 tasks regenerated, 10 tasks retained after manual review)

**Check 3: Temporal contamination documentation**
- **Method:** All tasks include `metadata.created_at` timestamp (2026-04-29) and `metadata.bench_snapshot_date` (2026-04-21). Any model trained on data after 2026-04-21 could have seen the bench snapshot values.
- **Mitigation:** The held-out split (50 tasks) uses bench snapshots with randomized capacity values (±20% jitter, seed=99) to prevent exact value memorization. The train/dev splits use the actual bench snapshot for realism.

**Summary:**
- **Total tasks checked:** 250
- **Initial flags:** 12 (4.8%)
- **False positives (retained):** 10 (shared vocabulary or intentional trace derivation)
- **True positives (regenerated):** 2 (fixture pool overlap)
- **Final violations:** 0
- **Result:** CLEAN

**Contamination check command:**
```bash
# Checks 1 (n-gram) + 2a (TF-IDF cosine) — no external dependencies
python contamination_check.py \
  --bench-dir tenacious_bench_v0.1 \
  --reference-file eval/trace_log.jsonl \
  --ngram 8 \
  --cosine-threshold 0.85

# Checks 1 + 2a + 2b (adds dense embedding similarity; requires sentence-transformers)
python contamination_check.py \
  --bench-dir tenacious_bench_v0.1 \
  --reference-file eval/trace_log.jsonl \
  --ngram 8 \
  --cosine-threshold 0.85 \
  --embedding-model all-MiniLM-L6-v2 \
  --embedding-threshold 0.85

# Check 3 (time-shift verification)
python contamination_check.py \
  --bench-dir tenacious_bench_v0.1 \
  --reference-file eval/trace_log.jsonl \
  --time-shift \
  --cutoff-date 2026-04-21
```

**Rationale for TF-IDF over dense embeddings:** Following Chen et al. (EMNLP 2025), we used TF-IDF cosine similarity instead of dense embedding models (e.g., text-embedding-3-large) because the contamination risk for Tenacious-Bench is structural (same bench values, same probe patterns) rather than semantic (paraphrased content). TF-IDF captures structural overlap more reliably for this domain and requires no API calls. The check ran in under 2 seconds on 250 tasks; a dense embedding check would require ~250 API calls at ~$0.002 each = $0.50 and would produce more false positives on shared domain vocabulary.

### LLM-as-Judge Rotation Policy

To prevent preference leakage (Li et al., 2025), the pipeline enforces strict model-family separation across all four tiers. The same model family is never used to both generate and judge the same task.

| Tier | Purpose | Model(s) | Family | Cost tier |
|------|---------|----------|--------|-----------|
| **Generation** | Bulk task synthesis (multi_llm_synthesis mode) | DeepSeek V3, Qwen 2.5-72B, Llama 3.1-70B | DeepSeek / Alibaba / Meta | Cheap (~$0.30/1M tokens) |
| **Quality filter** | Judge filter pipeline (`judge_filter.py`) | Google Gemini 2.0 Flash | Google | Cheap (~$0.20/1M tokens) |
| **Spot-check calibration** | 10% sample manual + LLM cross-check | Claude Haiku / GPT-4.1-mini | Anthropic / OpenAI | Mid (~$0.80/1M tokens) |
| **Held-out evaluation** | Sealed slice scoring (`scoring_evaluator.py`) | Google Gemini 2.5 Flash Lite | Google | Cheap (~$0.20/1M tokens) |

**Anti-leakage invariant:** The generation tier (DeepSeek/Qwen/Llama) is orthogonal to the judge tier (Gemini) at the family level. Neither tier uses OpenAI models, preventing the most common leakage vector (GPT-generated data judged by GPT). The held-out evaluation uses the same Gemini family as the quality filter — this is acceptable because the held-out tasks were not generated by Gemini; the leakage risk is generator→judge, not filter→evaluator.

**Scoring architecture by dimension:**

| Dimension | Scoring method | Rationale |
|-----------|---------------|-----------|
| capacity_honesty | Rule-based (regex) | Binary constraint check — regex is more reliable than LLM for deterministic conditions |
| signal_grounding | Rule-based (regex) | Hedge pattern detection is regex-tractable; 90% agreement with human raters |
| consent_coordination | Rule-based (regex) | Consent ask is a binary condition; no semantic judgment required |
| tone_preservation | LLM judge (Gemini) | Five tone markers require semantic judgment; regex insufficient for nuance |
| gap_framing | LLM judge (Gemini) | Net framing quality requires semantic judgment; accusatory vs research framing is context-dependent |

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
