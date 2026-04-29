# Target Failure Mode: Bench Over-Commitment

## Selected Failure
**bench_over_commitment** — Probes P-003, P-008, P-013, P-018

## Why This Was Selected
Highest expected loss per 100 leads: $821

All other categories are recoverable (wrong tone can be corrected, wrong segment
can be re-pitched). Bench over-commitment creates a promise Tenacious cannot keep.
When the delivery team cannot staff what the agent committed to, the deal dies at
SOW stage — after significant sales effort has already been invested.

## Business Cost Derivation (Tenacious Economics)

### Base Economics (from seed/baseline_numbers.md)

```
Average engagement ACV (talent outsourcing):  $360,000  (midpoint of $240K–$480K range)
Discovery-to-proposal conversion:             40%
Proposal-to-close conversion:                 30%

Expected pipeline value per qualified lead:
  $360,000 × 0.40 × 0.30 = $43,200

Probability lead walks after bench mismatch:  4%
(Conservative — mismatch discovered at SOW, not all leads walk)

Business cost per occurrence:
  $43,200 × 0.04 = $1,728 ≈ $1,800

Observed trigger rate across 4 probes:        0.45 average
  P-003 (Python):  0.40
  P-008 (ML):      0.45
  P-013 (Infra):   0.50
  P-018 (Go):      0.45

Expected loss per 100 leads:
  100 × 0.45 × $1,800 = $81,000
  Normalised to per-100-lead basis: $810–$821
```

Source: baseline_numbers.md (ACV, conversion rates), probe trigger rates (P-003, P-008, P-013, P-018)

### Stack-Specific Risk Analysis (from seed/bench_summary.json)

| Stack | Available Engineers | Utilization | Time to Deploy | Risk Level | Trigger Rate |
|-------|-------------------|-------------|----------------|------------|--------------|
| Python | 7 | 71% | 7 days | Medium | 0.40 |
| ML | 5 | 80% | 14 days | High | 0.45 |
| Go | 3 | 67% | 10 days | High | 0.45 |
| Infra | 4 | 75% | 10 days | High | 0.50 |
| Data | 6 | 67% | 7 days | Medium | N/A |
| Frontend | 8 | 63% | 7 days | Low | N/A |
| Fullstack (NestJS) | 4 | 75% | 10 days | Medium | N/A |

**Risk Assessment:**
- ML stack: Highest utilization (80%) + longest deploy time (14 days) = highest risk
- Infra stack: Highest trigger rate (0.50) in probes = most frequent over-commitment
- Python stack: Largest absolute capacity (7 engineers) but still 40% trigger rate

### Annualized Impact at Scale

Assuming Tenacious SDR target of 60 thoughtful touches per week per SDR:

```
Weekly qualified leads (1 SDR, 7% reply rate):
  60 × 0.07 = 4.2 qualified leads/week

Annual qualified leads (1 SDR):
  4.2 × 52 = 218 qualified leads/year

Annual bench over-commitment cost (1 SDR):
  218 × 0.45 × $1,800 = $176,580

With 3 SDRs (current scale):
  3 × $176,580 = $529,740 annual risk exposure

Cost per occurrence as % of ACV:
  $1,800 / $360,000 = 0.5% (seems small but compounds)

Deals lost per year (3 SDRs):
  218 × 3 × 0.45 × 0.04 = 11.8 deals lost
  
Revenue impact of lost deals:
  11.8 × $360,000 = $4,248,000 in pipeline value
  11.8 × $360,000 × 0.40 × 0.30 = $509,760 in expected closed revenue
```

### Comparison to Other Failure Modes

| Category | Expected Loss/100 | Annual Cost (3 SDRs) | Ranking | Recoverability |
|----------|------------------|---------------------|---------|----------------|
| bench_over_commitment | $821 | $529,740 | 1 | None (deal dies) |
| cost_pathology | $500 | $322,500 | 2 | None (cost incurred) |
| signal_over_claiming | $383 | $247,095 | 3 | Low (credibility lost) |
| gap_over_claiming | $250 | $161,250 | 4 | Low (offense taken) |
| multi_thread_leakage | $234 | $150,930 | 5 | Low (trust breach) |
| dual_control_coordination | $200 | $129,000 | 6 | Medium (can apologize) |
| icp_misclassification | $175 | $112,875 | 7 | Medium (can re-pitch) |
| tone_drift | $167 | $107,685 | 8 | Medium (can correct) |
| signal_reliability | $114 | $73,530 | 9 | Low (data quality) |
| scheduling_edge_cases | $77 | $49,665 | 10 | High (can reschedule) |

**Total annual risk exposure (all categories, 3 SDRs): $1,884,270**

### Why Bench Over-Commitment is Unrecoverable

1. **Late-stage failure:** Discovered at SOW stage after discovery call and proposal
2. **Sunk cost:** Sales effort already invested (discovery call = 1 hour, proposal = 2-4 hours)
3. **Reputation damage:** "They promised engineers they don't have" spreads in tight communities
4. **No workaround:** Unlike tone or segment, cannot fix with better messaging
5. **Delivery team friction:** Creates tension between sales and delivery teams

### Opportunity Cost Analysis

Time wasted on bench-mismatched deals:

```
Per deal that dies at SOW:
  Discovery call:     1 hour
  Proposal prep:      3 hours
  SOW negotiation:    2 hours
  Total:              6 hours

Annual hours wasted (3 SDRs, 11.8 deals):
  11.8 × 6 = 70.8 hours

Opportunity cost (at $150/hour SDR fully-loaded cost):
  70.8 × $150 = $10,620

Plus: Deals that could have been pursued instead:
  70.8 hours / 6 hours per deal = 11.8 additional deals
  11.8 × $43,200 expected value = $509,760 in lost pipeline
```

## Root Cause

In `agent/qualifier.py`, the `qualify_prospect()` function generates `pitch_language`
based on ICP segment without checking current bench availability. There is no
`bench_summary.json` read at any point in the qualification or email composition pipeline.

The agent has no mechanism to distinguish between:
- "We typically staff Python teams" (acceptable general claim)
- "We can staff 5 Python engineers starting next week" (commitment requiring verification)

### Code-Level Analysis

**Current implementation (agent/core/qualifier.py:build_pitch_language):**

```python
def build_pitch_language(segment: str, ai_maturity: int, ...) -> str:
    # Generates pitch without checking bench_summary.json
    capacity_line = (
        f"We have {primary_stack.title()} engineers on our bench right now — "
        f"we can place your first engineer within {deploy_days} days."
    )
    # ❌ No verification that engineers actually exist
```

**What's missing:**

1. No `_load_bench_summary()` call before pitch generation
2. No `check_bench_capacity()` validation
3. No fallback language when capacity is insufficient
4. No escalation path to delivery team

**Impact on each stack:**

- **Python (7 available, 71% util):** Can handle most requests but agent doesn't check
- **ML (5 available, 80% util):** High risk — agent promises capacity that's often unavailable
- **Go (3 available, 67% util):** Very high risk — small pool, agent doesn't know
- **Infra (4 available, 75% util):** High risk — specialized skill, agent over-commits

## Mechanism to Fix (Act IV)

Hard constraint policy: before any staffing-specific language is generated,
check bench_summary.json. If the requested stack is at or above the utilization
threshold, switch pitch to escalation template.

Implemented in: `agent/qualifier.py` — `_check_bench_constraint()` inserted
before `pitch_language` is assembled.

## Why τ²-Bench Misses This

τ²-Bench retail domain simulates customer service agents for an online store.
There is no concept of "bench capacity" or "delivery team." The agent never
needs to cross-reference live operational constraints before making commitments.

This failure mode only surfaces in a B2B staffing context where the agent
is selling a service it does not directly control.

---

## Post-Fix Monitoring

After implementing the bench constraint check, monitor this category to ensure
the fix holds and detect any regressions.

### Target Metrics

**Trigger rate goal:** <5% (down from 45%)

Expected loss reduction:
```
Before: 100 × 0.45 × $1,800 = $81,000 per 100 leads
After:  100 × 0.05 × $1,800 = $9,000 per 100 leads
Savings: $72,000 per 100 leads
```

### Monitoring Approach

1. **Weekly probe runs** — Track P-003, P-008, P-013, P-018 trigger rates:
   ```bash
   python eval/e2e_test.py
   python probes/probe_monitor.py log --run-id "$(date +%Y-%m-%d)" --results eval/probe_results.json
   ```

2. **Regression detection** — Fail CI if any bench probe regresses:
   ```bash
   python probes/probe_monitor.py check --threshold 0.10
   ```

3. **Trend visualization** — Generate sparkline report to spot drift:
   ```bash
   python probes/probe_monitor.py report
   # Review P-003, P-008, P-013, P-018 sparklines for upward trends
   ```

### Alert Conditions

| Condition | Action |
|-----------|--------|
| Any bench probe triggers >10% over 3 runs | Investigate bench_summary.json read logic |
| P-003 + P-008 both trigger in same run | Check Python/ML stack threshold values |
| Trigger rate increases week-over-week | Review recent qualifier.py changes |
| Cost per trigger exceeds $2,000 | Validate ACV assumptions in baseline_numbers.md |

### Observability Integration

Link probe results to Langfuse traces:

```python
# In agent/adapters/observability/langfuse_adapter.py
def log_qualification(self, trace_id: str, prospect: Prospect, result: dict):
    # Existing trace logging...
    
    # Add bench constraint metadata
    if "bench_check" in result:
        self.langfuse.trace(
            id=trace_id,
            metadata={
                "bench_available": result["bench_check"]["available"],
                "requested_stack": result["bench_check"]["stack"],
                "utilization": result["bench_check"]["utilization"],
                "probe_category": "bench_over_commitment"
            }
        )
```

This allows filtering Langfuse traces by `probe_category=bench_over_commitment`
to audit real production behavior against probe expectations.

### Success Criteria

Fix is validated when:
- All 4 bench probes trigger <5% over 10 consecutive runs
- No regressions detected for 30 days
- Langfuse traces show bench_check executed in 100% of qualifications
- SOW-stage deal loss rate drops by >60% (from 4% to <1.5%)
