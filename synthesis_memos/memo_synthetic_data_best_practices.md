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

## One-Line Disagreement for the Record

Liu et al. recommend LLM-as-judge for all quality filtering. For rule-based constraint dimensions (capacity_honesty, signal_grounding), regex-based scoring is more reliable and cheaper. The paper's recommendation applies to open-ended generation tasks; it does not generalize to constraint-following evaluation.
