# Synthesis Memo — A Survey on LLM-as-a-Judge
**Common Reading 4 | Gu et al., 2024–2025**

---

## Paper Summary

Gu et al. survey the design space of LLM-as-a-judge systems across three axes: **judge architecture** (single model vs panel vs cascade), **scoring protocol** (pointwise vs pairwise vs listwise), and **failure modes** (position bias, verbosity bias, self-enhancement bias, preference leakage). The paper's central finding: LLM judges are reliable for open-ended generation quality but unreliable for constraint-following evaluation — precisely because constraint-following requires checking a specific condition (did the agent check the bench?) rather than assessing overall quality.

The paper recommends: use LLM judges for dimensions that require semantic judgment (tone, framing, coherence); use rule-based scorers for dimensions that have verifiable ground truth (constraint satisfaction, format compliance).

---

## Application to Tenacious-Bench

### Where I agree

**Hybrid scoring is the right architecture.** The `scoring_evaluator.py` uses rule-based checks for `capacity_honesty` (regex for escalation language, bench snapshot check) and `signal_grounding` (regex for hedge patterns, forbidden assertion patterns), and reserves LLM-judge calls for `tone_preservation` and `gap_framing` where semantic judgment is required. This directly implements the paper's recommendation.

**Preference leakage is a real risk.** Li et al. (2025) — cited in the survey — document that using the same model family to generate training data and judge quality produces inflated quality scores. The Tenacious-Bench pipeline uses DeepSeek for bulk generation and reserves Claude/GPT-class models for spot-check calibration. This rotation policy prevents the same model from grading its own outputs.

**Pointwise scoring with explicit rubric dimensions is more reliable than holistic scoring.** The five tone markers in the Tenacious style guide map directly to five pointwise checks in `_check_tone_preservation()`. A holistic "is this email good?" prompt would be less reliable and harder to debug. The paper's finding that rubric-anchored pointwise scoring achieves higher inter-rater agreement than holistic scoring is confirmed by the 90% agreement rate in `inter_rater_agreement.md`.

### Where I disagree

**The paper recommends panel judges (multiple LLMs voting) for high-stakes evaluation.** For the Tenacious-Bench held-out evaluation, I used a single eval-tier model (Claude Sonnet or GPT-4.1 class) rather than a panel. The paper's recommendation is correct for general-purpose benchmarks where judge disagreement is informative. For Tenacious-Bench, the rubric is specific enough that a single well-calibrated judge produces consistent scores — the inter-rater agreement exercise confirmed this. A panel would triple the eval cost ($2–3 → $6–9) without meaningfully improving reliability.

**Evidence:** The three rubric disagreements in the inter-rater exercise were all resolved by rubric clarification, not by judge disagreement. The judges agreed on the rubric; the rubric was ambiguous. A panel of judges would have produced the same disagreements and required the same rubric fix. The fix was in the rubric, not in the judge architecture.

**The paper also recommends chain-of-thought prompting for LLM judges.** I agree this improves reliability for complex semantic judgments. However, for the `capacity_honesty` dimension, chain-of-thought prompting is unnecessary because the check is binary: did the agent check the bench before committing? A regex check is more reliable than a chain-of-thought LLM judge for this specific condition.

---

## Key Design Decisions Informed by This Paper

1. Hybrid scoring: rule-based for constraint dimensions, LLM judge for semantic dimensions.
2. Rotation policy: different model families for generation vs judging (preference leakage prevention).
3. Pointwise rubric-anchored scoring for all five dimensions.
4. Single eval-tier judge for held-out evaluation (cost-justified given rubric specificity).

---

## How I Operationalized My Alternative Design

My alternative to the paper's panel-judge recommendation: a single well-calibrated judge backed by a rubric-first calibration process.

- [x] **Rubric-first calibration before any judge calls.** `inter_rater_agreement.md` documents 30 tasks dual-labeled by two human raters. Three rubric ambiguities were identified and resolved *before* the LLM judge was deployed. This is the step the paper skips when recommending panels.
- [x] **Rule-based hard fails gate all LLM judge calls.** `scoring_evaluator.py` runs banned-phrase checks, condescending-pattern checks, and bench-external-ban checks before any API call. The LLM judge only sees outputs that have already passed the deterministic layer — reducing the semantic judgment surface to the genuinely ambiguous cases.
- [x] **Judge model is orthogonal to generator family.** `_DEFAULT_JUDGE_MODEL = "google/gemini-2.5-flash-lite"` in `scoring_evaluator.py`. Gemini is non-OpenAI, non-DeepSeek, non-Qwen — satisfying Li et al.'s anti-leakage requirement without a panel.
- [x] **Temperature=0 for determinism.** All judge calls use `"temperature": 0` in `_call_judge()`. A panel of three stochastic judges would introduce variance that a single deterministic judge avoids.
- [x] **Fallback to rule-based if judge unavailable.** `_tone_rule_based_soft()` is called when `_call_judge()` returns `None`. The evaluator never silently fails — it degrades gracefully to the deterministic layer.
- [x] **Inter-rater re-label after 24h confirms stability.** `inter_rater_agreement.md` §Re-Labeling documents 93.3% intra-rater agreement (κ=0.84), confirming the single-judge approach is stable over time without a panel.

---

## One-Line Disagreement for the Record

Gu et al. recommend panel judges for high-stakes evaluation. For a rubric-specific benchmark where judge disagreement traces to rubric ambiguity (not judge unreliability), a single well-calibrated judge is sufficient and three times cheaper. Fix the rubric first; add judges only if the rubric is genuinely ambiguous.
