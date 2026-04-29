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
4. Length difference must be ≤ 30% of the shorter output's word count.
5. Minimum 200 pairs total; 60% capacity_honesty, 20% signal_grounding, 20% gap_framing.

---

## Implementation Notes

- Pairs are generated programmatically using `training_data/generate_pairs.py`.
- Each pair is tagged with the source task_id, dimension, difficulty, and construction method.
- Quality gate: human spot-check of 20 pairs (10% sample) before training begins.
