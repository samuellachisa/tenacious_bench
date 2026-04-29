# Methodology Rationale — Path B (SimPO) for `bench_over_commitment`
**Act III Deliverable | One-Page Justification**

---

## The Problem, Stated Precisely

The Tenacious agent generates pitch language before performing any capacity check.
This produces the `bench_over_commitment` failure: the agent commits to staffing
(e.g. "we can place three ML engineers next sprint") before consulting
`bench_summary.json`. The expected loss is **$821 per 100 leads** — the highest
of any failure mode in the taxonomy (see `probes/failure_taxonomy.md`).

**The failure is not a knowledge gap.** The agent knows the bench check exists
(it has access to `bench_summary.json`). The failure is a *preference ordering*
defect: the model prefers the fluent, confident commitment over the cautious
escalation because fluent confidence is more common in its pretraining corpus.

---

## Why SFT Alone is Insufficient

Supervised Fine-Tuning (SFT) on gold demonstrations teaches the model to *produce*
the correct pattern. But for constraint-following failures, SFT introduces
reward hacking: the model learns to append "subject to confirmation" as a boilerplate
disclaimer while still making the hard commitment it was penalised for.
Zhou et al. (2023) document this in 15% of constraint-following SFT cases.
The model learns the surface form, not the underlying reasoning.

---

## Why SimPO is the Right Tool

SimPO (Meng et al., 2024) trains a preference over pairs (chosen, rejected)
where the chosen output genuinely exhibits the desired constraint behaviour and
the rejected output exhibits the failure mode. The loss directly penalises the
failure pattern at the token level — not just its absence in a gold demonstration.

**Three Week 10 evidence points that make SimPO appropriate here:**

1. **Trace `18725b79` (task_id=4, cost=$0.096):** The agent produced a 683-second,
   $0.096 multi-turn trace where it committed to an ML stack engagement in turn 1
   without any bench check. A SimPO rejected sample can be derived directly from
   this trace. The chosen sample is a rewrite with escalation language.

2. **Probe P-003 trigger rate = 0.45:** Nearly half of bench-relevant leads trigger
   this failure. The high trigger rate means the model has a strong prior toward
   the commitment pattern — strong enough that a preference signal, not just
   demonstration, is needed to shift it.

3. **Ablation Delta B (hard constraint vs soft warning, +1.0pp):** Even a
   rule-based hard constraint only moves the needle +1.0pp beyond a soft warning.
   This confirms the failure is in the model's generation preferences, not in
   access to information. Training is required; prompting is insufficient.

---

## Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Base model | Qwen3-8B (or Llama-3.1-8B-Instruct) | Fits on single A100 40GB |
| Training pairs | 200 (120 capacity_honesty, 40 signal_grounding, 40 gap_framing) | Per synthesis_memos/memo_pair_construction.md |
| Objective | SimPO (margin γ = 0.5, β = 2.0) | Reference-free; length-normalised |
| Epochs | 3 | Avoid overfitting on small pair set |
| Batch size | 4 (gradient accumulation 8) | Effective batch = 32 |
| Learning rate | 5e-6 with cosine decay | Conservative for small dataset |
| LoRA rank | 16 (α = 32) | Parameter-efficient; targets q/v projections |
| Evaluation cadence | Every 50 steps on dev split | Early stopping if pass@1 plateaus |
| Budget cap | $10 total (Unsloth free tier + local GPU) | Within project envelope |

---

## Success Criteria

The fine-tuned adapter is considered successful if:
- **Pass@1 on held-out split ≥ 77.67%** (current your_method: 74.67% + 3pp target).
- **Capacity honesty pass@1 ≥ 85%** (dimension-specific target).
- **No regression** on signal_grounding or tone_preservation (≤ 2pp drop).
- Loss converges within 3 epochs (no divergence).
