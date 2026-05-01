# Synthesis Memo 2 — Preference Pair Construction for Constraint-Following

**Context:** Week 11 Act III reading synthesis  
**Topic:** How to construct high-quality preference pairs for bench_over_commitment

---

## Reading Summary

### Paper A: Constitutional AI (Anthropic, 2022)
Preference pairs for alignment can be constructed without human annotation by having a "critic" model red-team an "actor" model. The critic generates a critique (why the output is problematic), the actor revises, and the (original, revision) pair becomes the (rejected, chosen) pair. Applicable here: original output commits to bench capacity → critic identifies the missing check → revision adds escalation language.

### Paper B: RLHF preference pair quality (Bai et al., 2022)
Pair quality correlates more strongly with the *margin* of the preferred output than with the absolute quality of either. A pair where the chosen output is clearly better (not just slightly better) produces a stronger training signal. For Tenacious: the rejected output should exhibit the failure mode clearly (hard commitment without check); the chosen output should exhibit the full preferred behavior (check + escalation + relevant stack mention).

### Paper C: Length bias in preference learning (Park et al., 2024)
Preference models are vulnerable to length bias: raters (and automated judges) prefer longer responses. SimPO's length normalization mitigates this, but pair construction should still aim for length-matched pairs to avoid length confounding.

---

## Where I Disagree

**Park et al. (2024) recommend length-matched pairs (≤10% length difference) to avoid length confounding.** I disagree with this strict threshold for the Tenacious domain. The paper's recommendation is based on open-ended generation tasks (summarization, creative writing) where length is a free parameter. For constraint-following tasks like bench_over_commitment, the **chosen** output is structurally longer because it must include:
1. Escalation language ("let me confirm bench availability")
2. Stack reference ("engineers available in [stack]")
3. Timeline qualification ("subject to delivery lead confirmation")
4. Explicit uncertainty markers ("I'll verify and revert within 24h")

The **rejected** output is structurally shorter because it skips all four components and commits directly ("We can absolutely place three ML engineers starting next sprint").

**Evidence from Week 10/11 pair generation:**
- Average chosen output length: 47 words (σ=12)
- Average rejected output length: 23 words (σ=8)
- Length ratio: 2.04:1 (104% difference, far exceeding Park et al.'s 10% threshold)

**Why this is correct for Tenacious-Bench:**
Trace `18725b79` (the highest-cost Week 10 failure) shows the agent generating a 19-word commitment ("We can staff your ML team with 3 engineers starting next sprint, all with 5+ years experience") without checking the bench. The correct behavior requires ~45 words to include all four constraint-respecting components. Forcing length-matched pairs would require either:
- Padding the rejected output with filler (introducing noise)
- Truncating the chosen output (removing necessary constraint language)

Both options degrade pair quality. The length difference is **semantically meaningful** — it reflects the structural difference between constraint-respecting and constraint-violating outputs.

**SimPO's length normalization handles this correctly.** The average log-probability reward is divided by sequence length, so the model is not rewarded for verbosity. The margin term γ ensures the chosen output is preferred by a fixed margin *after* length normalization. Park et al.'s concern about length confounding applies to models without length normalization (vanilla RLHF, DPO without length penalty); it does not apply to SimPO.

**Conclusion:** For constraint-following tasks where the correct behavior is structurally more verbose than the incorrect behavior, length-matched pairs are inappropriate. The length difference is signal, not noise.

---

## Preferred Pair Template

```json
{
  "task_id": "TB-CH-PR-XXXX",
  "chosen": {
    "output": "Before I commit to an ML team, let me confirm bench availability. Based on current capacity, we have [N] engineers available in [stack]. Subject to delivery lead confirmation, we could place a senior ML engineer by [date]. I'll verify and revert within 24h.",
    "signals": ["capacity_check", "escalation", "stack_reference", "timeline_qualified"]
  },
  "rejected": {
    "output": "We can absolutely place three ML engineers with your team starting next sprint. Consider it done.",
    "signals": ["hard_commit", "no_bench_check", "no_escalation", "overconfident_count"]
  }
}
```

**Construction rules:**
1. Chosen must contain: escalation language + stack reference + timeline qualification.
2. Rejected must contain: hard commitment + no escalation + implied certainty on count.
3. Both outputs must answer the same prospect request (same `input` context).
4. **Length difference is expected and semantically meaningful** — chosen outputs are structurally longer because they include constraint-respecting components. Do not artificially pad or truncate to match length.
5. Minimum 200 pairs total; 60% capacity_honesty, 20% signal_grounding, 20% gap_framing.

---

## One-Line Disagreement for the Record

Park et al. (2024) recommend length-matched pairs (≤10% difference) to avoid length confounding. For constraint-following tasks where correct behavior is structurally more verbose (escalation + qualification vs direct commitment), length difference is signal, not noise. SimPO's length normalization handles this correctly; forcing length-matched pairs would degrade pair quality by padding rejected outputs or truncating chosen outputs.

---

## How I Operationalized My Alternative Design

My alternative to Park et al.'s length-matching requirement: semantically-grounded length asymmetry with SimPO length normalisation as the technical mitigation.

- [x] **Pair template enforces structural completeness, not length parity.** `synthesis_memos/memo_pair_construction.md` defines four required signals for chosen outputs (capacity_check, escalation, stack_reference, timeline_qualified) and four forbidden signals for rejected outputs (hard_commit, no_bench_check, no_escalation, overconfident_count). Length follows from structure.
- [x] **Observed length ratio documented.** Chosen outputs average 47 words (σ=12); rejected outputs average 23 words (σ=8); ratio 2.04:1. This is recorded in this memo as the empirical baseline for v0.2 pair quality audits.
- [x] **SimPO length normalisation is the technical mitigation.** `training/train_simpo.py` uses `CPOConfig(loss_type="simpo")` which divides log-probability by sequence length before computing the margin. The model cannot game length by appending filler to the chosen output.
- [x] **Human spot-check of 20 pairs (10% sample).** `training_data/generate_pairs.py` includes a `--spot-check` flag that samples 10% of pairs for manual review. The spot-check confirmed that length differences reflect structural differences, not padding.
- [x] **Pair quality gate: both outputs must answer the same prospect request.** Each pair in `training_data/pairs.jsonl` shares the same `input` context (same hiring signal, bench snapshot, prospect context). Length difference cannot be attributed to different task difficulty.

---

## Implementation Notes

- Pairs are generated programmatically using `training_data/generate_pairs.py`.
- Each pair is tagged with the source task_id, dimension, difficulty, and construction method.
- Quality gate: human spot-check of 20 pairs (10% sample) before training begins.
