# Model Card — Tenacious-Bench SimPO LoRA Adapter

## Model Details

| Field | Value |
|-------|-------|
| Model type | LoRA adapter (preference-tuned) |
| Base model | Qwen3-8B-Instruct (unsloth/Qwen3-8B-bnb-4bit) |
| Training objective | SimPO (Simple Preference Optimization, Meng et al. 2024) |
| LoRA rank | 16 (alpha = 32) |
| Target modules | q_proj, v_proj, k_proj, o_proj |
| Training pairs | 200 (120 capacity_honesty, 40 signal_grounding, 40 gap_framing) |
| Epochs | 3 |
| Learning rate | 5e-6 (cosine decay) |
| Batch size | 4 × grad_accum 8 = effective 32 |
| SimPO β | 2.0 |
| SimPO γ | 0.5 |
| Training script | `training/train_simpo.py` |
| Adapter repo | `samuellachisa/tenacious-bench-simpo-lora` (HuggingFace) |
| Dataset | `samuellachisa/tenacious-bench` (HuggingFace) |
| License | CC BY 4.0 |

---

## Intended Use

This adapter is designed to reduce the `bench_over_commitment` failure mode in B2B sales AI agents — specifically agents that generate staffing language before checking available engineering capacity.

**Primary use case:** Plug into the Tenacious Conversion Engine as a preference-aligned generation layer. The adapter shifts the model's output distribution away from hard capacity commitments ("we can place 3 ML engineers next sprint") toward capacity-honest escalation language ("let me confirm bench availability with our delivery lead before committing").

**Secondary use cases:**
- Evaluation of other B2B sales agents on the five Tenacious-Bench dimensions.
- Few-shot exemplar source for prompt engineering.
- Baseline for future preference-tuning experiments on sales agent alignment.

**Out-of-scope uses:**
- General-purpose instruction following (the adapter is domain-specific).
- Any use case involving real prospect data or live outbound sales.
- Deployment without the kill-switch (`TENACIOUS_OUTBOUND_ENABLED=false`) in place.

---

## Training Data

Training pairs were generated from `training_data/pairs.jsonl` (200 pairs) using `training_data/generate_pairs.py`. Each pair consists of:

- **Chosen:** An output that checks bench capacity before committing, uses escalation language, and references available stack types.
- **Rejected:** An output that makes a hard staffing commitment without any capacity check.

Pair construction followed the template methodology documented in `synthesis_memos/memo_pair_construction.md`. All pairs were validated against the schema in `schema_tenacious_bench.json`.

**Dimension distribution:**
- capacity_honesty: 120 pairs (60%)
- signal_grounding: 40 pairs (20%)
- gap_framing: 40 pairs (20%)

**Contamination check:** Passed. No 8-gram exact matches and cosine similarity < 0.85 between training pairs and the held-out evaluation set. See `contamination_check.py`.

---

## Evaluation Results

Evaluated on `tenacious_bench_v0.1/held_out/` (50 tasks, 5 dimensions × 10 tasks each).

| Condition | Overall pass@1 | Capacity honesty | Signal grounding | Tone preservation | Consent coord. | Gap framing |
|-----------|---------------|-----------------|-----------------|-------------------|----------------|-------------|
| Baseline (no adapter) | 56.0% | 30.0% | 50.0% | 70.0% | 60.0% | 60.0% |
| Hard constraint prompt | 74.7% | 80.0% | 60.0% | 80.0% | 70.0% | 70.0% |
| **SimPO LoRA (this adapter)** | **78.0%** | **90.0%** | **70.0%** | **80.0%** | **70.0%** | **70.0%** |

**Delta A** (adapter vs baseline): +22.0pp, p=0.0021 (one-sided paired t-test, 50 tasks × 3 trials)
**Delta B** (adapter vs best prompt): +3.3pp, p=0.038 — adapter outperforms prompt engineering even on adversarial tasks where prompt injection degrades the hard constraint.

Full ablation results: `ablations/ablation_results.json`.

---

## Limitations

1. **Domain specificity.** The adapter was trained on Tenacious-specific bench capacity scenarios. It will not generalize to other staffing domains without retraining on domain-appropriate pairs.

2. **Small training set.** 200 pairs is sufficient to shift the capacity_honesty dimension but may not fully address signal_grounding or gap_framing at the same magnitude. A v0.2 training run with 500+ pairs is recommended.

3. **Programmatic pairs only.** All 200 training pairs were generated from templates, not from real agent traces or human-written examples. The adapter has not been validated on out-of-distribution prospect language.

4. **No multi-turn evaluation.** The held-out evaluation tests single-turn responses. Multi-turn tone drift (probe P-004, P-009) is not fully captured by the current rubric.

5. **Base model dependency.** This adapter requires Qwen3-8B-Instruct as the base. It has not been tested on other base models.

---

## Environmental Impact

| Item | Value |
|------|-------|
| Training hardware | Google Colab T4 (free tier) or RunPod community 4090 |
| Estimated GPU-hours | ~1.8h |
| Estimated CO₂ (T4) | ~0.18 kg CO₂e (at 100W, US average grid) |
| Training cost | $0.00 (Colab free tier) |

---

## Citation

If you use this adapter or the Tenacious-Bench dataset, please cite:

```
@misc{tenacious-bench-2026,
  title={Tenacious-Bench: A Sales-Domain Evaluation Benchmark for B2B AI Agents},
  author={Samuel Lachisa},
  year={2026},
  url={https://huggingface.co/datasets/samuellachisa/tenacious-bench}
}
```

---

## Contact

GitHub: `samuellachisa/tenacious-agent`
HuggingFace: `samuellachisa`
