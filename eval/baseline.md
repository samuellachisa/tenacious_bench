# τ²-Bench Retail Baseline — Tenacious Agent
**Act I Deliverable | 312 words**

## What Was Reproduced

The τ²-Bench retail domain baseline was provided by program staff via the
shared drive. Staff ran 5 trials × 30 tasks (150 simulations total) against
the retail development slice. All results are recorded in `eval/score_log.json`
and per-task traces in `eval/trace_log.jsonl` as provided by program staff.

The retail domain is the closest public analog to B2B qualification
conversation — multi-turn tasks with tool use and dual-control coordination
that mirror the Tenacious outbound pipeline.

All results are recorded in `eval/score_log.json` and per-task traces in
`eval/trace_log.jsonl` as provided by program staff.

## Results

| Metric | Value |
|--------|-------|
| Domain | retail |
| Tasks | 30 |
| Trials per task | 5 |
| Total simulations | 150 |
| Mean pass@1 | **0.7267 (72.67%)** |
| 95% CI lower | 0.6504 (65.04%) |
| 95% CI upper | 0.7917 (79.17%) |
| Avg agent cost per task | $0.0199 |
| p50 latency | 105.95 seconds |
| p95 latency | 551.65 seconds |
| Infrastructure errors | 0 |
| Git commit | d11a97072c49d093f7b5a3e4fe9da95b490d43ba |

## Comparison to Published Reference

| Source | pass@1 |
|--------|--------|
| τ²-Bench published reference (Feb 2026) | ~42% |
| This baseline (qwen3-next-80b) | **72.67%** |
| Delta | +30.67pp |

The gap reflects the stronger model used (qwen3-next-80b-a3b-thinking) versus
weaker dev-tier models in the published leaderboard.
This baseline is the zero point for Act IV improvement measurement.

## Unexpected Behavior

None. All simulations completed without infrastructure errors. Latency variance
was high (p50=106s vs p95=552s) — expected behavior for τ²-Bench retail with
complex multi-turn tool-call chains.

## Reproducibility

Results provided by program staff. To re-run independently:
```bash
pip install -r agent/requirements.txt
python eval/tau2_harness.py
# Model must be: openrouter/qwen/qwen3-next-80b-a3b-thinking
```