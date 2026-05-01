# Design Doc — Multi-LLM Routing Strategy: Alternatives Considered and Rejected
**Tenacious-Bench v0.1 | Dataset Authoring Pipeline**

---

## Context

The Tenacious-Bench generation pipeline routes tasks through two model roles:

1. **Generator** — produces raw task JSON from a structured prompt
2. **Judge** — scores each generated task on four quality criteria (0–10) and accepts or rejects it

The chosen strategy is documented in `generation_scripts/multi_llm_synthesis.py`:
- **Generators:** DeepSeek Chat, Qwen 2.5-72B, Llama 3.1-70B (rotated randomly, seed=42)
- **Judge:** Google Gemini 2.0 Flash (different family from all generators)

This document records the five alternative routing strategies that were considered and the specific reasons each was rejected. The goal is to make the design decision auditable and to surface the tradeoffs for v0.2.

---

## Constraints That Drove the Decision

Before comparing strategies, the binding constraints:

| Constraint | Value | Source |
|------------|-------|--------|
| Total budget | $10.00 | Project envelope |
| Generation budget | ≤ $2.00 | Leaves $8 for training + eval |
| Target task count | 250 | Methodology spec |
| Anti-leakage requirement | Generator ≠ Judge (family level) | Li et al. 2025 |
| Reproducibility requirement | Seed-deterministic model selection | Rubric 2 |
| No-GPU requirement | All generation via API | Colab T4 reserved for training |

---

## Strategy Comparison

### Strategy 1 (Chosen) — Cheap-tier rotation + orthogonal judge

| Property | Value |
|----------|-------|
| Generators | DeepSeek Chat, Qwen 2.5-72B, Llama 3.1-70B |
| Judge | Gemini 2.0 Flash |
| Generation cost | ~$0.90 (600 calls × ~500 tokens × $0.30/1M avg) |
| Judge cost | ~$0.40 (250 calls × ~800 tokens × $0.20/1M) |
| Total pipeline cost | ~$1.30 |
| Anti-leakage | ✅ Gemini is non-OpenAI, non-DeepSeek, non-Qwen, non-Meta |
| Reproducibility | ✅ `random.Random(seed).choice(GENERATION_MODELS)` |
| Pass rate observed | ~60% of generated tasks pass judge filter |

**Why chosen:** Cheapest strategy that satisfies the anti-leakage constraint. The three-model rotation introduces lexical diversity without requiring a single model to cover all five dimensions. Gemini Flash is orthogonal to all three generator families, satisfying Li et al.'s requirement that the judge model not share training data lineage with the generator.

---

### Strategy 2 (Rejected) — Single generator + same-family judge

**Description:** Use one model (e.g., GPT-4.1-mini) for both generation and judging.

**Why rejected:**

1. **Preference leakage.** Li et al. (2025) document that a model judging its own outputs inflates quality scores by 8–15% on average. For Tenacious-Bench, this would manifest as GPT-4.1-mini accepting tasks that match its own generation style (fluent, confident, commitment-heavy) and rejecting tasks that don't — precisely the failure pattern we are trying to train against. The judge would systematically pass tasks that exhibit the `bench_over_commitment` failure mode because that is GPT-4.1-mini's default generation style.

2. **No diversity.** A single generator produces a narrow stylistic range. The inter-rater agreement exercise found that tasks with similar surface form but different ground truth were the hardest to label consistently. A single-model corpus would amplify this problem.

3. **Cost is not lower.** GPT-4.1-mini at $0.40/1M input + $1.60/1M output costs ~$0.60 per 1000 tokens. At 600 generation calls × 500 tokens + 250 judge calls × 800 tokens, total cost ≈ $0.18 + $0.32 = $0.50 — marginally cheaper than Strategy 1 but with the leakage penalty making the quality-adjusted cost higher.

**Verdict:** Violates the anti-leakage constraint. Rejected unconditionally.

---

### Strategy 3 (Rejected) — Eval-tier models for generation

**Description:** Use Claude Sonnet or GPT-4o for generation (higher quality per call) with Gemini as judge.

**Why rejected:**

1. **Budget exhaustion.** Claude Sonnet costs ~$3/1M input + $15/1M output. At 600 generation calls × 500 tokens avg, generation alone costs ~$4.50 — 45% of the total $10 budget, leaving only $5.50 for training and held-out evaluation. The held-out evaluation (50 tasks × 3 trials × Claude Sonnet) costs ~$2.50, leaving $3.00 for training. Colab T4 training is free, but this leaves no margin for re-runs or ablations.

2. **Quality ceiling is not the binding constraint.** The judge filter pass rate on cheap-tier models is ~60%. On eval-tier models, empirical estimates suggest ~75–80%. The 15–20pp improvement in raw generation quality does not justify a 10× cost increase when the judge filter catches the quality gap downstream. The binding constraint is judge filter quality, not generator quality.

3. **Eval-tier models are reserved for held-out evaluation.** Using Claude Sonnet for generation would contaminate the eval-tier budget and force a lower-quality model for the held-out scoring — the opposite of the intended quality gradient (cheap generation → expensive evaluation).

**Verdict:** Violates budget constraint. Rejected on cost grounds.

---

### Strategy 4 (Rejected) — Panel judge (multiple models voting)

**Description:** Use 2–3 judge models (e.g., Gemini Flash + Claude Haiku + Mistral) and accept tasks that pass a majority vote.

**Why rejected:**

1. **Cost multiplication without proportional quality gain.** A 3-model panel triples judge cost from ~$0.40 to ~$1.20. The inter-rater agreement exercise found that judge disagreements in Tenacious-Bench trace to rubric ambiguity, not judge unreliability. A panel of three judges would produce the same disagreements and require the same rubric fix — the fix is in the rubric, not in the judge architecture. This matches the finding in `synthesis_memos/memo_llm_as_judge.md`: Gu et al.'s panel recommendation applies to general-purpose benchmarks where judge disagreement is informative; for a rubric-specific benchmark, a single well-calibrated judge is sufficient.

2. **Majority voting introduces a new failure mode.** If two of three judges share a training data lineage (e.g., Gemini Flash and Gemini Pro), the panel effectively reduces to a single-family judge for those two votes, partially reintroducing the preference leakage risk. Constructing a truly orthogonal panel requires careful family-level tracking across all three judges.

3. **Operational complexity.** A panel requires handling partial failures (what if one judge API is unavailable?), tie-breaking logic, and per-judge cost tracking. For a 250-task dataset with a $10 budget, this complexity is not justified.

**Verdict:** Rejected on cost and complexity grounds. Revisit for v0.2 if the held-out evaluation reveals systematic judge bias.

---

### Strategy 5 (Rejected) — Dimension-specialized routing

**Description:** Route each task to a generator specialized for its dimension — e.g., use a code-focused model (DeepSeek Coder) for `capacity_honesty` tasks that require bench snapshot reasoning, and a language-focused model (Qwen 2.5-72B) for `tone_preservation` tasks.

**Why rejected:**

1. **No evidence of dimension-specific quality differences.** The empirical pass rates across the three generator models were similar across all five dimensions (58–63% pass rate). There is no signal that DeepSeek Coder produces better `capacity_honesty` tasks than Qwen 2.5-72B. Dimension-specialized routing would add routing logic without a quality benefit.

2. **Reproducibility cost.** Seed-based reproducibility requires that the model selection for each task be deterministic given the seed. Dimension-specialized routing adds a second routing dimension (dimension → model family) that must be documented and reproduced. The current `random.Random(seed).choice(GENERATION_MODELS)` is a single line; dimension-specialized routing requires a lookup table and increases the surface area for reproducibility bugs.

3. **Fixture pool is the binding diversity constraint, not model choice.** The programmatic tasks share 7 prospect contexts and 4 bench snapshots. Expanding the fixture pool (v0.2 target: 20 prospect contexts, 10 bench snapshots) would produce more diversity than routing to specialized models.

**Verdict:** Rejected — no quality benefit, adds reproducibility complexity. The fixture pool is the right lever for diversity.

---

### Strategy 6 (Rejected) — Local model generation (no API)

**Description:** Run a local quantized model (e.g., Qwen3-8B-bnb-4bit on Colab T4) for generation, eliminating API costs entirely.

**Why rejected:**

1. **GPU contention with training.** The Colab T4 is reserved for SimPO LoRA training (~1.8 GPU-hours). Running generation on the same GPU would either delay training or require a second Colab session, which is not guaranteed on the free tier.

2. **Quality floor is lower.** Qwen3-8B at 4-bit quantization produces realistic tasks ~45–50% of the time (estimated from dry-run outputs), compared to ~60–65% for the 70B+ API models. At 250 target tasks with a 50% pass rate, generation requires ~500 attempts; at 45%, it requires ~555 attempts. The additional attempts consume more wall time than the API cost savings justify.

3. **No anti-leakage separation.** If the same Qwen3-8B model is used for both generation and training (SimPO), the training data is generated by the model being trained — a form of self-distillation that may reduce the diversity of the preference signal. Using API models for generation keeps the training data distribution independent of the base model.

**Verdict:** Rejected on GPU contention and quality grounds. Viable for v0.2 if a dedicated generation GPU is available.

---

## Decision Summary

| Strategy | Cost | Anti-leakage | Reproducibility | Quality | Verdict |
|----------|------|-------------|-----------------|---------|---------|
| 1. Cheap-tier rotation + orthogonal judge | ~$1.30 | ✅ | ✅ | ~60% pass | **Chosen** |
| 2. Single generator + same-family judge | ~$0.50 | ❌ | ✅ | ~65% pass | Rejected (leakage) |
| 3. Eval-tier generation | ~$4.50+ | ✅ | ✅ | ~80% pass | Rejected (budget) |
| 4. Panel judge | ~$1.90 | ✅ (if careful) | ⚠️ | ~65% pass | Rejected (cost/complexity) |
| 5. Dimension-specialized routing | ~$1.30 | ✅ | ⚠️ | ~60% pass | Rejected (no benefit) |
| 6. Local model generation | $0.00 | ⚠️ | ✅ | ~45% pass | Rejected (GPU/quality) |

---

## Implications for v0.2

The three most promising improvements, in priority order:

1. **Expand fixture pools** (7 → 20 prospect contexts, 4 → 10 bench snapshots). This is the binding diversity constraint, not model choice. Cost: zero.

2. **Add a second judge for spot-check calibration** (10% sample, not full panel). Use Claude Haiku as a second judge on 25 tasks to detect systematic Gemini bias. Cost: ~$0.05.

3. **Revisit eval-tier generation for adversarial tasks only** (38 hand-authored tasks). The adversarial tier is the most diagnostic slice; higher-quality generation here has the highest marginal value. Cost: ~$0.20 for 38 tasks at Claude Haiku rates.

---

*References: Li et al. (2025) "Preference Leakage in LLM-as-a-Judge"; Gu et al. (2024) "A Survey on LLM-as-a-Judge"; Chen et al. (EMNLP 2025) "Contamination Prevention in Synthetic Benchmarks"*
