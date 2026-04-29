# Test Description: Hiring Velocity Computation

## Function Under Test

`agent/core/enrichment.py::compute_hiring_velocity_label(current_count, historical_count)`

## Purpose

Computes a categorical velocity label and confidence score from current vs 60-day-ago job post counts. This label drives the agent's language in outreach emails (e.g., "tripled hiring" vs "modest growth" vs "ask rather than assert").

## Test Cases

### 1. Tripled or More Growth

```python
assert compute_hiring_velocity_label(12, 3) == ("tripled_or_more", 0.8)
assert compute_hiring_velocity_label(15, 4) == ("tripled_or_more", 0.8)
assert compute_hiring_velocity_label(100, 30) == ("tripled_or_more", 0.8)
```

**Rationale**: 3x+ growth is a strong hiring signal. Confidence is 0.8 when both counts are non-zero.

### 2. Doubled Growth

```python
assert compute_hiring_velocity_label(11, 4) == ("doubled", 0.8)
assert compute_hiring_velocity_label(8, 4) == ("doubled", 0.8)
assert compute_hiring_velocity_label(20, 10) == ("doubled", 0.8)
```

**Rationale**: 2x-3x growth. Example from sample brief: 11 today vs 4 sixty days ago = 2.75x = "doubled".

### 3. Increased Modestly

```python
assert compute_hiring_velocity_label(6, 5) == ("increased_modestly", 0.8)
assert compute_hiring_velocity_label(12, 10) == ("increased_modestly", 0.8)
assert compute_hiring_velocity_label(7, 4) == ("increased_modestly", 0.8)
```

**Rationale**: 1.2x-2x growth. Positive signal but not dramatic.

### 4. Flat (±20%)

```python
assert compute_hiring_velocity_label(10, 10) == ("flat", 0.8)
assert compute_hiring_velocity_label(10, 9) == ("flat", 0.8)
assert compute_hiring_velocity_label(10, 11) == ("flat", 0.8)
assert compute_hiring_velocity_label(5, 6) == ("flat", 0.8)
```

**Rationale**: 0.8x-1.2x ratio. No significant change in hiring activity.

### 5. Declined

```python
assert compute_hiring_velocity_label(5, 10) == ("declined", 0.8)
assert compute_hiring_velocity_label(2, 8) == ("declined", 0.8)
assert compute_hiring_velocity_label(0, 5) == ("declined", 0.6)
```

**Rationale**: <0.8x ratio. Hiring slowdown. When current is 0, confidence drops to 0.6.

### 6. Insufficient Signal (No Historical Data)

```python
assert compute_hiring_velocity_label(10, None) == ("insufficient_signal", 0.3)
assert compute_hiring_velocity_label(0, None) == ("insufficient_signal", 0.3)
assert compute_hiring_velocity_label(100, None) == ("insufficient_signal", 0.3)
```

**Rationale**: No 60-day snapshot available. Confidence is low (0.3). Agent must "ask rather than assert" about hiring velocity.

### 7. Edge Cases

```python
# Both zero
assert compute_hiring_velocity_label(0, 0) == ("flat", 0.6)

# Historical zero, current non-zero (can't compute ratio)
assert compute_hiring_velocity_label(1, 0) == ("increased_modestly", 0.6)
assert compute_hiring_velocity_label(3, 0) == ("tripled_or_more", 0.6)
assert compute_hiring_velocity_label(10, 0) == ("tripled_or_more", 0.6)

# Current zero, historical non-zero
assert compute_hiring_velocity_label(0, 10) == ("declined", 0.6)
```

**Rationale**: Division by zero handling. When historical is 0, infer growth magnitude from current count. Confidence drops to 0.6 for all edge cases.

## Integration Test

When integrated into `get_job_post_signals()`:

```python
# Mock scenario: Company had 4 roles 60 days ago, now has 11
job_signals = await get_job_post_signals("DataFlow Technologies", firmographics)

assert job_signals["open_roles"] == 11
assert job_signals["open_roles_60_days_ago"] == 4  # From historical snapshot
assert job_signals["velocity"] == "doubled"
assert job_signals["velocity_confidence"] == 0.8
```

## Current Limitation

The `open_roles_60_days_ago` field is currently hardcoded to `None` because no historical snapshot storage exists. When historical data is implemented:

1. Store job post counts in a time-series database (e.g., `data/job_snapshots/{company_domain}/{date}.json`)
2. Query the snapshot from 60 days ago in `get_job_post_signals()`
3. Pass the historical count to `compute_hiring_velocity_label()`
4. Confidence will automatically increase to 0.8 when historical data is available

## Schema Compliance

The output aligns with `schemas/hiring_signal_brief.schema.json`:

```json
{
  "hiring_velocity": {
    "open_roles_today": 11,
    "open_roles_60_days_ago": 4,
    "velocity_label": "doubled",
    "signal_confidence": 0.8,
    "sources": ["builtin", "wellfound", "company_careers_page"]
  }
}
```

Valid `velocity_label` values per schema:
- `tripled_or_more`
- `doubled`
- `increased_modestly`
- `flat`
- `declined`
- `insufficient_signal`

All labels are covered by the helper function.
