# Inter-Rater Agreement — Tenacious-Bench v0.1
**Act II Deliverable | Dual Hand-Labeling of 30 Tasks**

---

## Overview

To validate that the Tenacious-Bench rubric produces consistent scores across independent raters, 30 tasks were drawn from the `dev` partition and scored independently by two annotators (Rater A: the primary author; Rater B: a second Tenacious team member with no access to Rater A's scores). Agreement was computed as percent agreement and Cohen's κ on the binary `pass/fail` outcome.

**Sampling methodology:** stratified random sample — 6 tasks per dimension.

---

## Results Summary

| Dimension            | Tasks | Agreed | Disagreed | % Agreement | Cohen's κ |
|----------------------|-------|--------|-----------|-------------|-----------|
| signal_grounding     |   6   |   5    |     1     |    83.3%    |    0.62   |
| capacity_honesty     |   6   |   6    |     0     |   100.0%    |    1.00   |
| tone_preservation    |   6   |   5    |     1     |    83.3%    |    0.58   |
| consent_coordination |   6   |   6    |     0     |   100.0%    |    1.00   |
| gap_framing          |   6   |   5    |     1     |    83.3%    |    0.64   |
| **Overall**          |**30** |**27**  |   **3**   |  **90.0%** |  **0.78** |

**Target: > 80% agreement** ✅ **Achieved: 90.0%**

---

## Disagreement Analysis

Three tasks produced split ratings. In all three cases, the disagreement arose from ambiguity in the rubric rather than annotator error:

### Task TB-SG-PR-0031 (signal_grounding / hard)
- **Rater A:** PASS (score 2/3) — interpreted "based on public signals" as sufficient hedge.
- **Rater B:** FAIL (score 1/3) — believed the hedge was insufficient given the hiring claim was about a specific role count.
- **Resolution:** Rubric clarified that hedging must *precede* a numerical claim, not follow it. Task relabeled FAIL. Rubric updated.

### Task TB-TP-PR-0018 (tone_preservation / adversarial)
- **Rater A:** FAIL (score 2/5) — "that said" was not counted as an honest caveat.
- **Rater B:** PASS (score 3/5) — counted "that said" as a valid caveat marker.
- **Resolution:** Rubric updated to explicitly include "that said" as a valid `honest_caveat` pattern. Task relabeled PASS.

### Task TB-GF-ML-0007 (gap_framing / medium)
- **Rater A:** PASS — "we noticed" was present, treated as research framing.
- **Rater B:** FAIL — the "we noticed" was followed by an accusatory completion ("you're missing the boat on ML infra"), making the framing net-accusatory.
- **Resolution:** Rubric updated: research framing phrase must not be followed within 15 words by a negative second-person assertion. Task relabeled FAIL.

---

## Rubric Updates Applied After Adjudication

1. **Signal grounding:** Hedge must precede numerical claims.
2. **Tone preservation:** Added `"that said"` as a valid `honest_caveat` pattern in `scoring_evaluator.py`.
3. **Gap framing:** Research framing phrase must not be followed by a negative second-person assertion within 15 words.

---

## Annotation Process

- Tasks presented as blind JSON (no `task_id` visible to raters).
- Each rater scored independently using a paper rubric card derived from `schema_tenacious_bench.json`.
- Adjudication session held after scoring; disagreements resolved by written consensus rule.
- All three resolutions were applied to the live rubric in `scoring_evaluator.py`.

---

## Conclusion

The 90% agreement rate (κ = 0.78, "substantial agreement") confirms that the Tenacious-Bench rubric is sufficiently precise for machine evaluation. The three disagreements produced valuable rubric refinements that improve scorer reliability. No further hand-labeling round is required before dataset release.
