# Test Description: Bench Capacity Constraint

## Functions Under Test

- `agent/core/qualifier.py::check_bench_capacity(required_stack, required_count)`
- `agent/core/qualifier.py::infer_required_stacks(enrichment)`
- `agent/core/qualifier.py::build_pitch_language()` (capacity-aware pitch generation)

## Purpose

Prevents the highest-ROI failure mode (bench_over_commitment): agent promising engineering capacity that exceeds bench_summary.json counts. This addresses Probes P-003, P-008, P-013, P-018 which collectively cost $821 per 100 leads.

## Root Cause (Fixed)

Previously, `build_pitch_language()` generated pitch text without checking bench_summary.json. Agent would say "We have strong ML engineering capability" even when ML bench showed 0 available engineers.

## Solution

1. `check_bench_capacity()`: Loads bench_summary.json and compares required vs available counts
2. `infer_required_stacks()`: Infers which stacks prospect needs from job signals and AI maturity
3. `build_pitch_language()`: Integrates capacity check and adjusts language accordingly

## Test Cases

### 1. Sufficient Capacity (Happy Path)

```python
# Bench has 7 Python engineers, prospect needs 1
capacity = check_bench_capacity("python", 1)

assert capacity["available"] == True
assert capacity["available_count"] == 7
assert capacity["required_count"] == 1
assert capacity["gap"] == 6
assert "7 Python engineers available" in capacity["recommendation"]
assert "within 7 days" in capacity["recommendation"]
```

**Expected pitch language**: "We have 7 Python engineers on our bench right now — we can place your first engineer within 7 days."

### 2. Insufficient Capacity (Phased Ramp)

```python
# Bench has 5 ML engineers, prospect needs 6
capacity = check_bench_capacity("ml", 6)

assert capacity["available"] == False
assert capacity["available_count"] == 5
assert capacity["required_count"] == 6
assert capacity["gap"] == -1
assert "5 engineers available" in capacity["recommendation"]
assert "ramp the remaining 1 within 2-3 weeks" in capacity["recommendation"]
```

**Expected pitch language**: "Our ML bench currently has 5 engineers available. We can start with 5 and ramp the remaining 1 within 2-3 weeks — would that timeline work for your needs?"

### 3. Zero Capacity (Escalation Required)

```python
# Bench has 0 Go engineers (hypothetical), prospect needs 2
capacity = check_bench_capacity("go", 2)

assert capacity["available"] == False
assert capacity["available_count"] == 0
assert capacity["gap"] == -2
assert "currently at capacity" in capacity["recommendation"]
assert "delivery lead" in capacity["recommendation"]
```

**Expected pitch language**: "Our Go bench is currently at capacity. Let me connect you with our delivery lead to discuss timeline and alternatives."

### 4. Stack Inference from AI Maturity

```python
enrichment = {
    "ai_maturity": {"score": 2, "confidence": "high"},
    "job_signals": {"ai_roles": ["ML Engineer", "Data Scientist"]},
    "firmographics": {"open_roles_raw": ["ML Engineer", "Data Scientist", "Backend Engineer"]}
}

stacks = infer_required_stacks(enrichment)

assert "ml" in stacks  # AI maturity >= 2 triggers ML stack
assert "python" in stacks  # Backend Engineer triggers Python
```

### 5. Stack Inference from Job Posts

```python
enrichment = {
    "ai_maturity": {"score": 0},
    "job_signals": {"ai_roles": []},
    "firmographics": {
        "open_roles_raw": [
            "Senior Data Engineer (dbt, Snowflake)",
            "DevOps Engineer (Kubernetes, Terraform)",
            "Frontend Engineer (React, TypeScript)"
        ]
    }
}

stacks = infer_required_stacks(enrichment)

assert "data" in stacks  # dbt, Snowflake keywords
assert "infra" in stacks  # Kubernetes, Terraform keywords
assert "frontend" in stacks  # React, TypeScript keywords
```

### 6. Integration Test: Pitch Language with Capacity Check

```python
enrichment = {
    "company": "DataFlow Technologies",
    "ai_maturity": {"score": 2, "confidence": "high"},
    "job_signals": {
        "open_roles": 3,
        "ai_roles": ["ML Engineer"],
    },
    "firmographics": {
        "open_roles_raw": ["ML Engineer", "Data Engineer", "Backend Engineer"]
    },
    "funding_event": {
        "type": "series_b",
        "total_funding_usd": 14_000_000,
        "days_ago": 45,
        "in_window": True
    }
}

pitch = build_pitch_language("recently_funded", 2, "high", enrichment)

# Should include capacity check for ML stack (primary inferred stack)
assert "ML engineers" in pitch or "engineers available" in pitch
assert "within" in pitch  # Deployment timeline mentioned
assert "bench" in pitch.lower()  # Explicit bench reference

# Should NOT promise capacity that doesn't exist
# If bench has 5 ML engineers, should say "5 ML engineers available"
# NOT "we can staff any size team" or similar unbounded claims
```

## Probe Coverage

This fix addresses the following high-severity probes:

### P-003: Python Over-Commitment
- **Before**: "We can get your Python team started within 2 weeks" (when bench has 2, prospect needs 5)
- **After**: "Our Python bench currently has 2 engineers available. We can start with 2 and ramp the remaining 3 within 2-3 weeks — would that timeline work?"

### P-008: ML Over-Commitment
- **Before**: "We have strong ML engineering capability" (when ML bench = 0)
- **After**: "Our ML bench is currently at capacity. Let me connect you with our delivery lead to discuss timeline and alternatives."

### P-013: Infra Over-Commitment
- **Before**: "Our infra team has strong Kubernetes experience" (no capacity check)
- **After**: "We have 4 Infra engineers on our bench right now — we can place your first engineer within 14 days."

### P-018: Go Over-Commitment
- **Before**: "We have experienced Go engineers available" (no capacity check)
- **After**: "We have 3 Go engineers on our bench right now — we can place your first engineer within 14 days."

## Business Impact

- **Expected loss reduction**: $821 per 100 leads → ~$0 (assuming 100% fix effectiveness)
- **Failure mode**: Delivery mismatch kills deal and damages reputation
- **Fix confidence**: High — hard constraint check before pitch generation
- **Monitoring**: Log `bench_capacity_check` trace event to Langfuse for every pitch

## Limitations

1. **Single stack inference**: Currently checks only the primary inferred stack. If prospect needs multiple stacks (e.g., ML + Data + Infra), only the first is validated.

2. **No multi-engineer validation**: If prospect explicitly asks for "5 Python engineers", the system checks capacity for 1 engineer by default. Need to parse prospect's message for explicit count requests.

3. **No utilization target check**: bench_summary.json includes `utilization_target_pct` (e.g., 75% for Python). Current implementation doesn't factor this in — if Python has 7 available but target is 75%, effective capacity is ~5.

4. **Static snapshot**: bench_summary.json is updated weekly. Real-time capacity may differ if engineers are placed between updates.

## Next Steps

1. Add Langfuse trace event for every capacity check: `log_trace("bench_capacity_check", capacity_result)`

2. Extend to multi-stack validation: Check all inferred stacks, not just primary

3. Parse prospect messages for explicit count requests: "Can you staff 5 Python engineers?" → `required_count=5`

4. Factor in utilization targets: `effective_capacity = available_count * (1 - utilization_target_pct / 100)`

5. Add weekly bench snapshot validation: Alert if bench_summary.json is >7 days stale
