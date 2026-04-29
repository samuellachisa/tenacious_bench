# Failure Taxonomy — Tenacious Agent

## Category Summary

| Category | Probes | Avg Trigger Rate | Avg Business Cost | Expected Loss / 100 leads |
|----------|--------|-----------------|-------------------|--------------------------|
| bench_over_commitment | P-003, P-008, P-013, P-018 | 0.45 | $1,825 | **$821** |
| signal_over_claiming | P-001, P-006, P-011, P-016, P-021, P-026 | 0.55 | $697 | **$383** |
| gap_over_claiming | P-027, P-028 | 0.25 | $1,000 | **$250** |
| tone_drift | P-004, P-009, P-014 | 0.25 | $667 | **$167** |
| icp_misclassification | P-002, P-007, P-012, P-017 | 0.39 | $450 | **$175** |
| signal_reliability | P-023, P-024, P-025 | 0.18 | $633 | **$114** |
| multi_thread_leakage | P-005, P-010, P-015 | 0.22 | $1,067 | **$234** |
| dual_control_coordination | P-029 | 0.40 | $500 | **$200** |
| scheduling_edge_cases | P-019, P-020, P-022 | 0.22 | $350 | **$77** |
| cost_pathology | P-030 | 0.10 | $5,000 | **$500** |

---

## Ranked by Expected Loss per 100 Leads

1. **bench_over_commitment** — $821 (Probes P-003, P-008, P-013, P-018)
2. **signal_over_claiming** — $383 (Probes P-001, P-006, P-011, P-016, P-021, P-026)
3. **cost_pathology** — $500 (Probe P-030) ← note: low frequency but catastrophic at scale
4. **gap_over_claiming** — $250 (Probes P-027, P-028)
5. **dual_control_coordination** — $200 (Probe P-029)
6. **icp_misclassification** — $175 (Probes P-002, P-007, P-012, P-017)
7. **multi_thread_leakage** — $234 (Probes P-005, P-010, P-015)
8. **tone_drift** — $167 (Probes P-004, P-009, P-014)
9. **signal_reliability** — $114 (Probes P-023, P-024, P-025)
10. **scheduling_edge_cases** — $77 (Probes P-019, P-020, P-022)

---

## Category Detail

### bench_over_commitment (Highest ROI — Selected as Target)
Agent promises engineering capacity without checking bench_summary.json.
Affects all four stack types: Python, ML, Go, Infra.
Root cause: qualifier.py generates pitch_language before any capacity check.
Fix: Hard constraint check before staffing language is generated.

### signal_over_claiming
Agent asserts facts it cannot verify from public data.
Most common: hiring velocity claimed without historical baseline, AI maturity
inferred from marketing copy, stale news presented as current.
Root cause: enrichment.py returns signals without confidence gates on output language.
Fix: Confidence-aware phrasing — assert only when confidence=high, ask when medium/low.

### gap_over_claiming
Competitor gap framing sounds accusatory rather than advisory.
Root cause: build_competitor_gap_brief() gap language is written as statements not questions.
Fix: Reframe as "potential opportunity" not "you're falling behind."

### cost_pathology
Playwright retry loops with no ceiling create runaway LLM + scraping costs.
Root cause: _scrape_careers_page() has no max_retries limit.
Fix: max_retries=2, fallback to cached data, hard $0.10 per-interaction budget cap.
