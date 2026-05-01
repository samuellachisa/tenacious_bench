# Synthesis Memo — Best Practices and Lessons Learned on Synthetic Data
**Common Reading 1 | Liu et al., COLM 2024**

---

## Paper Summary

Liu et al. survey the landscape of synthetic data generation for LLM training and evaluation. The core argument: synthetic data quality is determined by three properties — **diversity** (covering the input space), **accuracy** (ground truth is correct), and **difficulty** (tasks are hard enough to be informative). The paper identifies five failure modes: distribution collapse (all synthetic examples look the same), label noise (LLM-generated ground truth is wrong), contamination (synthetic data overlaps with eval), reward hacking (model learns surface patterns not underlying behavior), and annotation inconsistency (rubric applied differently across examples).

---

## Application to Tenacious-Bench

### Where I agree

**Diversity via parameter sweeps is necessary but not sufficient.** The paper's finding that programmatic sweeps produce high lexical diversity but low semantic diversity matches what I observed in the Tenacious-Bench generation. The 250 tasks cross 5 dimensions × 4 difficulties × 4 source modes, but the fixture pools (7 prospect contexts, 4 bench snapshots) limit the semantic range. A v0.2 dataset should expand the fixture pools or use LLM-generated prospect contexts.

**Accuracy of ground truth is the binding constraint.** The inter-rater agreement exercise (90%, κ=0.78) confirmed this directly: three of the 30 hand-labeled tasks had ambiguous ground truth, not ambiguous candidate outputs. The rubric was the source of noise, not the tasks. This matches Liu et al.'s finding that annotation inconsistency is the most common quality failure in synthetic benchmarks.

**Difficulty stratification matters.** The paper shows that easy tasks contribute little signal to model evaluation — they are passed by all models and provide no discrimination. The Tenacious-Bench adversarial tier (50 tasks, 20% of total) is the most diagnostic slice. The ablation results confirm this: the baseline model passes 70% of easy tasks but only 30% of adversarial tasks.

### Where I disagree

**The paper recommends LLM-as-judge for all quality filtering.** I disagree for the Tenacious domain. The five tone markers in the Tenacious style guide are specific enough that a rule-based scorer (regex + keyword checks) is more reliable than an LLM judge for the `capacity_honesty` and `signal_grounding` dimensions. An LLM judge introduces preference leakage risk (Li et al., 2025) and adds cost. I used LLM judges only for `tone_preservation` and `gap_framing`, where the rubric requires semantic judgment that regex cannot capture.

**Evidence:** The scoring_evaluator.py achieves 90% agreement with human raters on capacity_honesty using pure regex — no LLM judge needed. This is a specific design choice that diverges from Liu et al.'s recommendation and is justified by the domain's rule-based nature.

---

## Key Design Decisions Informed by This Paper

1. **Four authoring modes** (trace-derived, programmatic, multi-LLM synthesis, hand-authored) directly implement the paper's recommendation for diverse generation pipelines.
2. **Inter-rater agreement at 80% threshold** follows the paper's quality gate recommendation.
3. **Contamination check** (n-gram + cosine) implements the paper's contamination prevention protocol.
4. **Adversarial tier** implements the paper's difficulty stratification recommendation.

---

## How I Operationalized My Alternative Design

My alternative to Liu et al.'s LLM-as-judge-for-all-filtering recommendation: a hybrid scorer that uses rule-based checks for constraint dimensions and reserves LLM judges for semantic dimensions only.

- [x] **Rule-based scoring for capacity_honesty and signal_grounding.** `scoring_evaluator.py` `_check_capacity_honesty()` and `_check_signal_grounding()` use regex patterns loaded from `style_guide_config.md`. No LLM call is made for these dimensions unless `--llm-judge` is explicitly passed.
- [x] **LLM judge reserved for tone_preservation and gap_framing.** `_check_tone_preservation()` and `_check_gap_framing()` call `_call_judge()` only when `_LLM_JUDGE_ENABLED` is True. The judge prompt is rubric-anchored (five explicit tone markers, 0–5 scale) not holistic.
- [x] **90% inter-rater agreement on capacity_honesty with pure regex.** `inter_rater_agreement.md` confirms the rule-based scorer matches human raters at 90% (κ=0.78) on capacity_honesty — the dimension Liu et al. would route to an LLM judge. This is the empirical evidence that regex is sufficient for constraint dimensions.
- [x] **Preference leakage prevented by judge model rotation.** `scoring_evaluator.py` `_DEFAULT_JUDGE_MODEL = "google/gemini-2.5-flash-lite"` — a different family from the generation models (DeepSeek, Qwen, Llama). This directly addresses Li et al.'s leakage risk that Liu et al. underweight.
- [x] **Graceful fallback when judge is unavailable.** `_tone_rule_based_soft()` is called when `_call_judge()` returns `None`. The evaluator never silently fails — it degrades to the deterministic layer and logs a warning.
- [x] **Cost difference is documented.** `ablations/ablation_results.json` `cost_quality_analysis` shows rule-based scoring costs $0.000354/task vs $0.000816/task with LLM judge — a 130% cost increase for dimensions where regex achieves equivalent accuracy.

---

## One-Line Disagreement for the Record

Liu et al. recommend LLM-as-judge for all quality filtering. For rule-based constraint dimensions (capacity_honesty, signal_grounding), regex-based scoring is more reliable and cheaper. The paper's recommendation applies to open-ended generation tasks; it does not generalize to constraint-following evaluation.
