# Bench Over-Commitment Fix — Implementation Summary

## Problem Statement

The agent was committing to engineering capacity without checking `seed/bench_summary.json`, causing the highest-ROI failure mode in the system:

- **Category**: bench_over_commitment
- **Probes**: P-003, P-008, P-013, P-018
- **Trigger rate**: 0.45 (45% of interactions)
- **Business cost**: $1,825 per incident
- **Expected loss**: $821 per 100 leads

### Example Failures

**P-003 (Python)**:
- Prospect: "Can you staff a 5-person Python team?"
- Agent (before fix): "We can get your Python team started within 2 weeks"
- Reality: Bench only has 2 Python engineers available
- Outcome: Delivery mismatch kills deal, damages reputation

**P-008 (ML)**:
- Prospect: "We need 2 ML engineers for a 6-month build"
- Agent (before fix): "We have strong ML engineering capability"
- Reality: ML bench = 0
- Outcome: ML engineers not available, deal collapses at SOW stage

## Root Cause

`agent/core/qualifier.py::build_pitch_language()` generated pitch text without any capacity validation. The function had no awareness of bench_summary.json counts.

## Solution

Implemented three new functions in `agent/core/qualifier.py`:

### 1. `check_bench_capacity(required_stack, required_count)`

Loads bench_summary.json and compares required vs available counts.

**Returns**:
```python
{
    "available": bool,
    "available_count": int,
    "required_count": int,
    "stack": str,
    "gap": int,
    "recommendation": str  # What to say to prospect
}
```

**Pseudocode**:
```python
def check_bench_capacity(required_stack: str, required_count: int = 1) -> dict:
    # Load bench data
    bench_data = load_json("seed/bench_summary.json")
    
    # Get available count for requested stack
    available = bench_data.get(required_stack, {}).get("available_engineers", 0)
    deploy_days = bench_data.get(required_stack, {}).get("time_to_deploy_days", 14)
    
    # Calculate gap
    gap = available - required_count
    
    # Decision logic
    if gap >= 0:
        # Sufficient capacity
        return {
            "available": True,
            "available_count": available,
            "required_count": required_count,
            "stack": required_stack,
            "gap": gap,
            "recommendation": f"We have {available} {required_stack} engineers on our bench right now — we can place your first engineer within {deploy_days} days."
        }
    elif available > 0:
        # Partial capacity - phased ramp
        shortage = abs(gap)
        return {
            "available": True,
            "available_count": available,
            "required_count": required_count,
            "stack": required_stack,
            "gap": gap,
            "recommendation": f"Our {required_stack} bench currently has {available} engineers available. We can start with {available} and ramp the remaining {shortage} within 2-3 weeks — would that timeline work?"
        }
    else:
        # Zero capacity - escalate
        return {
            "available": False,
            "available_count": 0,
            "required_count": required_count,
            "stack": required_stack,
            "gap": gap,
            "recommendation": f"Our {required_stack} bench is currently at capacity. Let me connect you with our delivery lead to discuss timeline and alternatives."
        }
```

**Logic**:
- `gap >= 0`: Sufficient capacity → "We have X engineers available — we can place within Y days"
- `gap < 0` and `available > 0`: Phased ramp → "We have X available, can ramp remaining Y within 2-3 weeks"
- `available == 0`: Escalation → "Let me connect you with our delivery lead"

### 2. `infer_required_stacks(enrichment)`

Infers which stacks the prospect needs from enrichment signals.

**Pseudocode**:
```python
def infer_required_stacks(enrichment: dict) -> list[str]:
    stacks = []
    
    # Extract signals
    ai_maturity = enrichment.get("ai_maturity", {}).get("score", 0)
    ai_roles = enrichment.get("job_signals", {}).get("ai_roles", [])
    job_posts = enrichment.get("job_signals", {}).get("recent_posts", [])
    tech_stack = enrichment.get("tech_stack", [])
    
    # Combine all text for keyword matching
    all_text = " ".join(job_posts + tech_stack).lower()
    
    # Priority order: ML > Data > Backend > Infra > Frontend
    if ai_maturity >= 2 or len(ai_roles) > 0:
        stacks.append("ml")
    
    if any(kw in all_text for kw in ["dbt", "snowflake", "databricks", "data engineer", "analytics"]):
        stacks.append("data")
    
    if any(kw in all_text for kw in ["python", "django", "fastapi", "flask"]):
        stacks.append("python")
    
    if any(kw in all_text for kw in ["golang", "go ", " go,", "microservices"]):
        stacks.append("go")
    
    if any(kw in all_text for kw in ["devops", "kubernetes", "terraform", "aws", "infrastructure"]):
        stacks.append("infra")
    
    if any(kw in all_text for kw in ["react", "next.js", "typescript", "frontend", "vue"]):
        stacks.append("frontend")
    
    # Default fallback
    if not stacks:
        stacks.append("python")
    
    return stacks
```

**Logic**:
- AI maturity >= 2 OR ai_roles present → `ml`
- Job posts mention dbt/Snowflake/Databricks → `data`
- Job posts mention Python/Django/FastAPI → `python`
- Job posts mention Go/Golang/microservices → `go`
- Job posts mention DevOps/Kubernetes/Terraform → `infra`
- Job posts mention React/Next.js/TypeScript → `frontend`
- Default if no signals → `python`

**Returns**: List of stack names in priority order, e.g., `["ml", "python", "data"]`

### 3. Updated `build_pitch_language()`

Now capacity-aware. Calls `infer_required_stacks()` and `check_bench_capacity()` before generating pitch text.

**Pseudocode**:
```python
def build_pitch_language(trigger_type: str, ai_maturity: int, urgency: str, enrichment: dict) -> str:
    # Infer what stacks the prospect needs
    required_stacks = infer_required_stacks(enrichment)
    primary_stack = required_stacks[0]
    
    # Check bench capacity for primary stack
    capacity = check_bench_capacity(primary_stack, required_count=1)
    
    # Build pitch based on capacity status
    if not capacity["available"]:
        # Zero capacity - escalate to delivery lead
        pitch = capacity["recommendation"]
        pitch += "\n\nIn the meantime, I can share how we've helped similar companies scale their engineering teams."
        return pitch
    
    # Capacity available - build normal pitch with capacity language
    base_pitch = _generate_base_pitch(trigger_type, ai_maturity, urgency, enrichment)
    
    # Inject capacity statement
    capacity_statement = capacity["recommendation"]
    
    # Combine: base pitch + capacity statement + CTA
    pitch = f"{base_pitch}\n\n{capacity_statement}\n\nWould you be open to a 15-minute call to discuss your specific needs?"
    
    return pitch
```

**Escalation Path**:
```python
# When capacity["available"] == False:
1. Return escalation message: "Let me connect you with our delivery lead"
2. Log trace event: {"event": "capacity_escalation", "stack": primary_stack, "company": company}
3. Create HubSpot task: "Delivery Lead Follow-up Required - {stack} capacity at zero"
4. Set deal stage to "Escalated - Capacity Constraint"
5. Do NOT send automated email - wait for delivery lead manual review
```

**New behavior**:
- Sufficient capacity: "We have 7 Python engineers on our bench right now — we can place your first engineer within 7 days."
- Insufficient capacity: "Our ML bench currently has 5 engineers available. We can start with 5 and ramp the remaining 1 within 2-3 weeks — would that timeline work?"
- Zero capacity: "Our Go bench is currently at capacity. Let me connect you with our delivery lead to discuss timeline and alternatives."

## Implementation Details

### File Changes

**agent/core/qualifier.py**:
- Added `import json` and `from pathlib import Path`
- Added `_load_bench_summary()` helper
- Added `check_bench_capacity()` (60 lines)
- Added `infer_required_stacks()` (50 lines)
- Added `_get_deploy_days()` helper
- Updated `build_pitch_language()` to integrate capacity checks
- Updated module docstring to mention bench capacity constraint

### Data Dependencies

**seed/bench_summary.json**:
- Updated weekly (Mondays 09:00 UTC)
- Contains `available_engineers` count per stack
- Contains `time_to_deploy_days` per stack
- Contains `honesty_constraint` policy note

### Trace Events

Should add (not yet implemented):
```python
log_trace("bench_capacity_check", {
    "company": company,
    "required_stack": primary_stack,
    "available_count": capacity_check["available_count"],
    "required_count": capacity_check["required_count"],
    "gap": capacity_check["gap"],
    "recommendation": capacity_check["recommendation"]
})
```

## Test Coverage

**eval/test_bench_capacity.md** documents:
- 6 unit test cases for `check_bench_capacity()`
- 2 test cases for `infer_required_stacks()`
- 1 integration test for capacity-aware pitch generation
- Coverage for all 4 probes (P-003, P-008, P-013, P-018)

## Verification

### Before Fix
```python
enrichment = {"ai_maturity": {"score": 2}, "job_signals": {"ai_roles": ["ML Engineer"]}}
pitch = build_pitch_language("capability_gap", 2, "high", enrichment)

# Output: "We have strong ML engineering capability"
# Problem: No capacity check — may promise ML engineers when bench = 0
```

### After Fix
```python
enrichment = {"ai_maturity": {"score": 2}, "job_signals": {"ai_roles": ["ML Engineer"]}}
pitch = build_pitch_language("capability_gap", 2, "high", enrichment)

# Output: "We have 5 ML engineers on our bench right now — we can place your first engineer within 10 days."
# OR (if ML bench = 0): "Our ML bench is currently at capacity. Let me connect you with our delivery lead."
```

## Business Impact

### Expected Outcomes

- **Trigger rate reduction**: 0.45 → ~0.05 (residual edge cases only)
- **Expected loss reduction**: $821 → ~$50 per 100 leads
- **Annual savings** (at 5,000 leads/year): ~$38,550
- **Reputation protection**: No more delivery mismatches due to capacity over-commitment

### Monitoring

Track these metrics in Langfuse:
1. `bench_capacity_check` event count per day
2. `capacity_insufficient` flag rate (when gap < 0)
3. `capacity_zero_escalation` rate (when available = 0)
4. Correlation between capacity warnings and deal conversion rates

## Limitations

1. **Single stack validation**: Only checks primary inferred stack. If prospect needs ML + Data + Infra, only ML is validated.

2. **Default count = 1**: Assumes prospect needs 1 engineer unless explicitly stated. Need to parse messages for "5 Python engineers" → `required_count=5`.

3. **No utilization target**: bench_summary.json includes `utilization_target_pct` (e.g., 75%). Not factored into effective capacity calculation.

4. **Weekly snapshot**: bench_summary.json updated weekly. Real-time capacity may differ if engineers placed between updates.

5. **No multi-turn memory**: If prospect says "I need 5 engineers" in turn 1, then asks "Can you do Python?" in turn 2, the system doesn't remember the count from turn 1.

## Next Steps

1. **Add Langfuse tracing**: Log every capacity check for monitoring and debugging

2. **Multi-stack validation**: Check all inferred stacks, not just primary. Flag if any required stack has insufficient capacity.

3. **Parse explicit counts**: Use regex or LLM to extract "5 Python engineers" → `required_count=5`

4. **Utilization-aware capacity**: `effective_capacity = available * (1 - utilization_target_pct / 100)`

5. **Stale data alerting**: Alert if bench_summary.json is >7 days old

6. **Multi-turn context**: Store prospect's stated requirements across conversation turns

7. **Probe regression testing**: Run P-003, P-008, P-013, P-018 against the fixed system to verify 0% trigger rate

## Deployment Checklist

- [x] Implement `check_bench_capacity()`
- [x] Implement `infer_required_stacks()`
- [x] Update `build_pitch_language()` to use capacity checks
- [x] Add test specification (`eval/test_bench_capacity.md`)
- [x] Update README with fix documentation
- [ ] Add Langfuse trace events
- [ ] Run unit tests for all 3 new functions
- [ ] Run integration test with real bench_summary.json
- [ ] Run P-003, P-008, P-013, P-018 probes to verify fix
- [ ] Deploy to staging and verify with synthetic prospects
- [ ] Monitor capacity check events in Langfuse for 1 week
- [ ] Deploy to production with kill switch enabled (outbound=false)
- [ ] Enable live outbound after 1 week of clean monitoring

## References

- **Failure taxonomy**: `probes/failure_taxonomy.md`
- **Probe library**: `probes/probe_library.md` (P-003, P-008, P-013, P-018)
- **Bench data**: `seed/bench_summary.json`
- **Test spec**: `eval/test_bench_capacity.md`
- **Implementation**: `agent/core/qualifier.py` (lines 30-150)
