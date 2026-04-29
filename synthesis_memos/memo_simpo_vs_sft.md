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
