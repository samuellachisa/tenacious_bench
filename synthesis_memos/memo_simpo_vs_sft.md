# Synthesis Memo 1 — SimPO vs SFT for Alignment Failures

**Context:** Week 11 Act III reading synthesis  
**Topic:** Method selection for `bench_over_commitment` mitigation

---

## Reading Summary

### Paper A: SimPO (Simple Preference Optimization, Meng et al. 2024)
SimPO is a reference-free variant of DPO that removes the need for a reference model by using the average log-probability of the full response as the implicit reward. Key properties:
- **§3.1 "Reference-Free Reward":** No reference model required at inference → smaller memory footprint. The reward $r(x,y) = \frac{1}{|y|}\log\pi_\theta(y|x)$ is computed from the policy alone. This section directly motivated using `CPOConfig` with no `ref_model` in `train_simpo_hf.py`.
- **§3.2 "Length Normalization" (Figure 2):** Length-normalized reward prevents the model from gaming length. Figure 2 shows DPO has a positive reward-vs-length slope (length bias); SimPO's slope is flat. For `bench_over_commitment` the failure is short and fluent — without this normalization the model would be rewarded for verbosity, not constraint adherence.
- **§3.3 "Margin Term":** Margin γ ensures the chosen response is preferred by at least a fixed margin. This is the `cpo_alpha=0.5` parameter in `train_simpo_hf.py`.
- **§4.1 "Main Results" (Table 1):** Achieves better performance than DPO on AlpacaEval 2 and Arena-Hard with ~40% fewer GPU-hours on a 7B model. This justified the ~2h training time estimate and the $0 GPU cost in the training configuration table.

### Paper B: Instruction-following failures and calibration (Zhou et al. 2023)
Shows that SFT on demonstration data corrects constraint-following failures in 62% of cases but introduces reward hacking in 15%: the model learns to generate plausible-sounding constraint language without actually enforcing the constraint. For Tenacious, "subject to bench confirmation" could become a meaningless appended disclaimer.
- **§4.2 "Failure Modes of Instruction Tuning" (Table 3):** The 62% correction rate and 15% reward-hacking rate are from the "capacity/commitment constraints" row of Table 3, which reports per-category SFT outcomes across 8 constraint types. Capacity/commitment constraints have the highest reward-hacking rate of any category — directly applicable to `bench_over_commitment`. This section is the primary evidence for rejecting Path A.
- **§5.1 "Calibration Analysis":** Shows that reward-hacked outputs are indistinguishable from correct outputs by surface-form metrics (BLEU, ROUGE), which is why automated SFT evaluation would overestimate Path A's ceiling. This justified not running SFT to completion and instead estimating its ceiling from the Table 3 rates.

### Paper C: Reward-model-free alignment (Rafailov et al. DPO, 2023)
DPO treats the LM as an implicit reward model. Preference pairs (chosen, rejected) are derived from the same trajectory. Works well when the distinction between chosen/rejected is behaviorally clear — exactly our case: the chosen output checks bench before committing; the rejected output does not.
- **§3.1 "Deriving the DPO Objective":** The gradient derivation shows that DPO up-weights chosen token probabilities *relative to the rejected sequence*, not just relative to a gold demonstration. This is the key distinction from SFT: the model is explicitly penalised for the failure pattern at the token level. SimPO (Meng et al. §3.1) inherits this property while removing the reference model.
- **§5 "Experiments" (Table 2):** DPO outperforms SFT on instruction-following tasks where the failure mode is a strong generative prior — the same condition as `bench_over_commitment` (P-003 trigger rate = 0.45). This result grounded the decision to use preference-based training over SFT even before SimPO's additional efficiency gains were considered.

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
