# Synthesis Memo — Datasheets for Datasets + Data Cards
**Common Reading 2 | Gebru et al. 2021 + Pushkarna et al. FAccT 2022**

---

## Paper Summaries

### Datasheets for Datasets (Gebru et al., 2021)
Proposes a standardized documentation format for datasets covering seven sections: motivation, composition, collection process, preprocessing/cleaning/labeling, uses, distribution, and maintenance. The core argument: undocumented datasets cause downstream harm because users cannot assess fitness for purpose. The datasheet is a communication artifact between dataset creators and consumers.

### Data Cards (Pushkarna et al., FAccT 2022)
Extends Datasheets with a layered detail model: **telescopic** (one-paragraph summary for quick assessment), **periscopic** (section-level detail for informed use), **microscopic** (field-level provenance for auditing). The key addition over Datasheets is the explicit acknowledgment that different consumers need different levels of detail — a practitioner deploying a model needs different information than a researcher auditing bias.

---

## Application to Tenacious-Bench

### Where I agree

**The seven Gebru sections are necessary.** The `datasheet.md` covers all seven. The most important section for Tenacious-Bench is **Uses** — specifically the warning that the benchmark is grounded in Tenacious-specific business rules (bench_summary.json format, style guide) and will require adaptation for other sales agents. Without this warning, a user could apply the benchmark to a different agent and get misleading scores.

**Layered detail is genuinely useful.** The Pushkarna telescopic/periscopic/microscopic model maps well to the Tenacious-Bench documentation structure:
- Telescopic: the README quickstart (one paragraph, headline numbers).
- Periscopic: the datasheet.md (section-level detail on composition, collection, uses).
- Microscopic: the individual task JSON files (field-level provenance via `metadata.author_model`, `metadata.judge_score`, `metadata.seed`).

### Where I disagree

**Gebru et al. treat the datasheet as a static artifact.** For a living benchmark like Tenacious-Bench (v0.1 → v0.2 planned), the datasheet needs versioning. The paper does not address how to handle datasheet updates when the dataset changes. I added a `Maintenance` section to `datasheet.md` that explicitly documents the v0.2 roadmap and the contamination-check requirement for each version increment. This is a practical extension the paper does not cover.

**Pushkarna et al. recommend separate Data Cards for each dataset split.** For a 250-task benchmark with three partitions, this would produce three near-identical cards. I disagree: the held-out partition's card should be embargoed (not published until evaluation completes), but the train and dev partitions share the same provenance. A single datasheet with a partition table is more maintainable and less likely to drift out of sync.

**Evidence for disagreement:** The inter-rater agreement exercise produced three rubric revisions that would have required updating three separate Data Cards under Pushkarna's recommendation. A single datasheet required one update. Maintenance cost is a real constraint for a small team.

---

## Key Design Decisions Informed by These Papers

1. `datasheet.md` follows all seven Gebru sections.
2. Each task JSON includes `metadata.created_at`, `metadata.author_model`, `metadata.judge_score` — microscopic provenance per Pushkarna.
3. The held-out partition is embargoed from public release — per Gebru's distribution section guidance.
4. The `Maintenance` section documents the v0.2 roadmap and contribution process.

---

## One-Line Disagreement for the Record

Pushkarna et al. recommend separate Data Cards per split. For a small benchmark with shared provenance across splits, a single versioned datasheet with a partition table is more maintainable and less likely to produce documentation drift.
