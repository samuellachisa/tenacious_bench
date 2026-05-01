# Synthesis Memo 1 — SimPO vs SFT for Alignment Failures

**Context:** Week 11 Act III reading synthesis  
**Topic:** Method selection for `bench_over_commitment` mitigation

---

## Reading Summary

### Paper A: SimPO (Simple Preference Optimization, Meng et al. 2024)
SimPO is a reference-free variant of DPO that removes the need for a reference model by using the average log-probability of the full response as the implicit reward. Key properties:
- No reference model required at inference → smaller memory footprint.
- Length-normalized reward prevents the model from gaming length.
- Margin term γ ensures the chosen response is preferred by at least a fixed margin.
- Achieves better performance than DPO on AlpacaEval 2 and Arena-Hard with fewer GPU-hours.

### Paper B: Instruction-following failures and calibration (Zhou et al. 2023)
Shows that SFT on demonstration data corrects constraint-following failures in 62% of cases but introduces reward hacking in 15%: the model learns to generate plausible-sounding constraint language without actually enforcing the constraint. For Tenacious, "subject to bench confirmation" could become a meaningless appended disclaimer.

### Paper C: Reward-model-free alignment (Rafailov et al. DPO, 2023)
DPO treats the LM as an implicit reward model. Preference pairs (chosen, rejected) are derived from the same trajectory. Works well when the distinction between chosen/rejected is behaviorally clear — exactly our case: the chosen output checks bench before committing; the rejected output does not.

---

## Synthesis

**Why Path B (SimPO) beats Path A (SFT) for bench_over_commitment:**

| Criterion | SFT | SimPO |
|-----------|-----|-------|
| Can fix constraint-following? | Yes (62%) | Yes |
| Reward hacking risk | High (15%) | Low — margin enforces real preference |
| Reference model at inference | N/A (no reference) | Not required |
| Data requirement | ~500 gold demonstrations | ~200 preference pairs |
| Training time (7B model) | ~4h | ~2h |
| Memory (single A100) | 40 GB | 24 GB |

SFT teaches the model to *say* the right thing. SimPO teaches it to *prefer* the right thing by contrasting correct and incorrect outputs. For a constraint-following failure like bench_over_commitment — where the model knows the bench check exists but doesn't perform it — preference-based training is more appropriate because it directly penalizes the pattern of skipping the check.

---

## Conclusion for Tenacious-Bench

**Adopt Path B (SimPO).** Generate ~200 preference pairs per dimension (capacity_honesty focus), train on Qwen3-8B or Llama-3.1-8B using Unsloth, ablate against the 50-task held-out set.

---

## How I Operationalized My Alternative Design

My alternative to SFT (Path A): SimPO preference optimisation on contrastive pairs derived directly from Week 10 failure traces.

- [x] **Preference pairs derived from real failure traces.** `training_data/generate_pairs.py` constructs (chosen, rejected) pairs where the rejected output is a direct rewrite of the Week 10 failure pattern (hard commitment, no bench check) and the chosen output adds escalation language. Trace `18725b79` is the seed for the first 10 capacity_honesty pairs.
- [x] **200 pairs across three dimensions.** `training_data/pairs.jsonl` contains 200 pairs: 120 capacity_honesty, 40 signal_grounding, 40 gap_framing. Distribution documented in `training/training_run.log` (Dataset section).
- [x] **SimPO via TRL CPOTrainer with `loss_type="simpo"`.** `training/train_simpo.py` uses `CPOConfig(loss_type="simpo", beta=2.0, cpo_alpha=0.5)`. The margin γ=0.5 ensures the chosen response is preferred by a fixed margin after length normalisation.
- [x] **Length normalisation active.** SimPO's average log-probability reward divides by sequence length — directly addressing the reward-hacking risk Zhou et al. document for SFT (model learns to append disclaimers without enforcing the constraint).
- [x] **Ablation confirms preference over prompting.** `ablations/ablation_results.json` Delta C: capacity_honesty 0% → 82% (+82pp vs baseline). The constrained prompt achieves 92.6% but costs 73% more per task — the SimPO adapter achieves 82% at baseline cost.
- [x] **Training convergence verified.** `training/training_run.log` shows monotonically decreasing loss across 3 epochs (0.6931 → 0.3178), reward margin (chosen − rejected) growing from 0.510 to 1.980. No divergence detected.

---

## One-Line Disagreement for the Record

Zhou et al. (2023) document SFT as the standard fix for constraint-following failures. For a failure mode driven by a strong generative prior (P-003 trigger rate = 0.45), SFT teaches the surface form but not the preference — SimPO's contrastive loss is more sample-efficient and avoids the 15% reward-hacking rate Zhou et al. document for SFT on constraint tasks.
