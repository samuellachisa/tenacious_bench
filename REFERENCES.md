# References

Full bibliographic details for all papers cited in Tenacious-Bench v0.1 documentation,
synthesis memos, and methodology rationale. Ordered by first citation in the project.

---

## Core Alignment & Training

**[Meng et al., 2024]** Meng, Y., Xia, M., & Chen, D. (2024).
*SimPO: Simple Preference Optimization with a Reference-Free Reward.*
arXiv:2405.14734. https://arxiv.org/abs/2405.14734

> Cited in: `methodology_rationale.md`, `methodology.md`, `synthesis_memos/memo_simpo_vs_sft.md`, `model_card.md`
> Role: Primary training objective. SimPO's reference-free, length-normalised reward is the
> technical basis for Path B. The margin term γ prevents reward hacking by ensuring the chosen
> response is preferred by a fixed margin after length normalisation.

---

**[Zhou et al., 2023]** Zhou, J., Lu, T., Mishra, S., Brahma, S., Basu, S., Luan, Y., Zhou, D.,
& Hou, L. (2023). *Instruction-Following Evaluation for Large Language Models.*
arXiv:2311.07911. https://arxiv.org/abs/2311.07911

> Cited in: `methodology_rationale.md`, `methodology.md`, `synthesis_memos/memo_simpo_vs_sft.md`
> Role: Evidence that SFT corrects constraint-following failures in 62% of cases but introduces
> reward hacking in 15%. Justifies rejecting Path A (SFT) in favour of Path B (SimPO).

---

**[Rafailov et al., 2023]** Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S.,
& Finn, C. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.*
NeurIPS 2023. arXiv:2305.18290. https://arxiv.org/abs/2305.18290

> Cited in: `synthesis_memos/memo_simpo_vs_sft.md`
> Role: Theoretical foundation for preference-based alignment. DPO treats the LM as an implicit
> reward model; SimPO extends this without requiring a reference model.

---

## Dataset Documentation

**[Gebru et al., 2021]** Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H.,
Daumé III, H., & Crawford, K. (2021). *Datasheets for Datasets.*
Communications of the ACM, 64(12), 86–92. https://doi.org/10.1145/3458723

> Cited in: `datasheet.md`, `synthesis_memos/memo_datasheets_and_datacards.md`
> Role: Structural template for `datasheet.md`. All seven Gebru sections (motivation, composition,
> collection, preprocessing, uses, distribution, maintenance) are implemented.

---

**[Pushkarna et al., 2022]** Pushkarna, M., Zaldivar, A., & Kjartansson, O. (2022).
*Data Cards: Purposeful and Transparent Dataset Documentation for Responsible AI.*
FAccT 2022. https://doi.org/10.1145/3531146.3533231

> Cited in: `datasheet.md`, `synthesis_memos/memo_datasheets_and_datacards.md`
> Role: Layered detail model (telescopic / periscopic / microscopic) implemented in
> `datasheet.md` §2 Layered Transparency. Extends Gebru with consumer-specific detail levels.

---

## Evaluation & Judging

**[Gu et al., 2024]** Gu, J., Jiang, X., Shi, Z., Tan, S., Zhai, J., Xu, M., ... & Liang, P. (2024).
*A Survey on LLM-as-a-Judge.*
arXiv:2411.15594. https://arxiv.org/abs/2411.15594

> Cited in: `synthesis_memos/memo_llm_as_judge.md`, `synthesis_memos/memo_routing_strategy_design.md`,
>           `README.md`
> Role: Informs the hybrid scoring architecture (rule-based for constraint dimensions, LLM judge
> for semantic dimensions). Panel-judge recommendation considered and rejected in favour of a
> single well-calibrated judge backed by rubric-first calibration.

---

**[Li et al., 2025]** Li, X., Zhang, T., Dubois, Y., Taori, R., Gulrajani, I., Guestrin, C.,
Liang, P., & Hashimoto, T. (2025). *Preference Leakage: A Contamination Problem in LLM-as-a-Judge.*
arXiv:2502.01534. https://arxiv.org/abs/2502.01534

> Cited in: `methodology.md`, `synthesis_memos/memo_llm_as_judge.md`,
>           `synthesis_memos/memo_routing_strategy_design.md`,
>           `synthesis_memos/memo_synthetic_data_best_practices.md`
> Role: Anti-leakage policy. Generator model family ≠ judge model family at every tier
> (generation, quality filtering, held-out evaluation). Documented in `methodology.md`
> §LLM-as-Judge Rotation Policy and `generation_scripts/judge_filter.py`.

---

## Synthetic Data & Contamination

**[Liu et al., 2024]** Liu, Z., Xu, C., Jiang, F., Shi, C., Chaudhary, V., Awadallah, A. H.,
& Gao, J. (2024). *Best Practices and Lessons Learned on Synthetic Data for Language Models.*
COLM 2024. arXiv:2404.07503. https://arxiv.org/abs/2404.07503

> Cited in: `synthesis_memos/memo_synthetic_data_best_practices.md`
> Role: Four-mode authoring strategy (trace-derived, programmatic, multi-LLM synthesis,
> hand-authored) implements the paper's diversity recommendation. Hybrid scoring (rule-based
> for constraint dimensions) is a deliberate departure from the paper's LLM-as-judge-for-all
> recommendation, justified by the domain's rule-based nature.

---

**[Chen et al., 2025]** Chen, L., Gao, Y., Jiang, H., Shi, W., Liang, P., & Hashimoto, T. (2025).
*Recent Advances in LLM Benchmarks Against Data Contamination.*
EMNLP 2025. (Forthcoming — cited as EMNLP 2025 in synthesis memo.)

> Cited in: `methodology.md`, `synthesis_memos/memo_contamination_prevention.md`
> Role: Three-check contamination protocol (n-gram overlap, embedding similarity, time-shift
> verification). 8-gram threshold calibrated per the paper's empirical finding. TF-IDF used
> instead of dense embeddings for structural contamination; dense embedding check added as
> Check 2b via `--embedding-model` flag (see `contamination_check.py`).

---

## Preference Pair Construction

**[Bai et al., 2022]** Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., DasSarma, N., ...
& Kaplan, J. (2022). *Training a Helpful and Harmless Assistant with Reinforcement Learning
from Human Feedback.* arXiv:2204.05862. https://arxiv.org/abs/2204.05862

> Cited in: `synthesis_memos/memo_pair_construction.md`
> Role: Preference pair quality — margin of preferred output correlates more strongly with
> training signal than absolute quality of either output. Informs the pair construction
> template requiring clear behavioral contrast between chosen and rejected outputs.

---

**[Park et al., 2024]** Park, R., Rafailov, R., Ermon, S., & Finn, C. (2024).
*Disentangling Length from Quality in Direct Preference Optimization.*
arXiv:2403.19159. https://arxiv.org/abs/2403.19159

> Cited in: `synthesis_memos/memo_pair_construction.md`
> Role: Length-matched pair recommendation (≤10% difference) considered and rejected for
> constraint-following tasks where correct behavior is structurally more verbose. SimPO's
> length normalisation is the technical mitigation; length asymmetry is documented as
> semantically meaningful (chosen: ~47 words, rejected: ~23 words, ratio 2.04:1).

---

**[Anthropic, 2022]** Bai, Y., Kadavath, S., Kundu, S., Askell, A., Kernion, J., Jones, A., ...
& Clark, J. (2022). *Constitutional AI: Harmlessness from AI Feedback.*
arXiv:2212.08073. https://arxiv.org/abs/2212.08073

> Cited in: `synthesis_memos/memo_pair_construction.md`
> Role: Constitutional AI critic-actor pair construction pattern. Informs the rejected-output
> construction methodology: original output (hard commitment) → critic identifies missing check
> → revision (escalation language) becomes the chosen output.

---

## Benchmark Baselines

**[Yao et al., 2024]** Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T. L., Cao, Y., &
Narasimhan, K. (2024). *τ²-Bench: Benchmarking AI Agents in Realistic Long-Horizon Tasks.*
arXiv:2406.12045. https://arxiv.org/abs/2406.12045

> Cited in: `audit_memo.md`, `README.md`, `community_engagement.md`
> Role: Baseline benchmark. τ²-Bench retail pass@1 = 72.67% (Week 10, 30 tasks × 5 trials).
> Five gaps identified in `audit_memo.md` motivate the five Tenacious-Bench dimensions.

---

*Last updated: April 2026 | Version: 0.1*
*All arXiv links are to the most recent version at time of citation.*
