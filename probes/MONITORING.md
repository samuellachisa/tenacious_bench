# Probe Monitoring Guide

The probe monitoring tool tracks trigger rates over time, visualizes trends, and detects regressions.

## Quick Start

### 1. Log a probe run

After running your evaluation suite, log the results:

```bash
python probes/probe_monitor.py log \
  --run-id "2026-04-25-baseline" \
  --results probes/example_probe_results.json
```

The results JSON should map probe IDs to trigger status:

```json
{
  "P-001": {
    "triggered": true,
    "cost": 847,
    "notes": "Still over-claiming hiring signal"
  },
  "P-002": {
    "triggered": false,
    "cost": 0,
    "notes": "Mixed signal logic fixed"
  }
}
```

### 2. Generate trend report

Create an HTML visualization of probe trends:

```bash
python probes/probe_monitor.py report --output probes/trigger_trends.html
```

Open `probes/trigger_trends.html` in a browser to see:
- Current trigger status for all probes
- Sparkline showing last 10 runs
- Trend direction (improving 📉 or worsening 📈)
- Historical trigger rate

### 3. Check for regressions

Run this in CI to fail the build if probes regress:

```bash
python probes/probe_monitor.py check --threshold 0.15
```

Exit codes:
- `0` = No regressions detected
- `1` = Regressions found (probe that passed before now triggers)

## Workflow Integration

### Daily Development

```bash
# After making changes, run evaluation
python eval/e2e_test.py

# Log results
python probes/probe_monitor.py log \
  --run-id "$(date +%Y-%m-%d)-dev" \
  --results eval/probe_results.json

# Check for regressions
python probes/probe_monitor.py check
```

### Weekly Review

```bash
# Generate trend report
python probes/probe_monitor.py report

# Open in browser
open probes/trigger_trends.html

# Review high-cost probes that are still triggering
# Prioritize fixes based on business cost
```

### CI/CD Pipeline

```yaml
# .github/workflows/probe-check.yml
name: Probe Regression Check

on: [push, pull_request]

jobs:
  probe-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Run evaluation
        run: python eval/e2e_test.py
      
      - name: Log probe results
        run: |
          python probes/probe_monitor.py log \
            --run-id "${{ github.sha }}" \
            --results eval/probe_results.json
      
      - name: Check for regressions
        run: python probes/probe_monitor.py check
      
      - name: Upload trend report
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: probe-trends
          path: probes/trigger_trends.html
```

## History Format

Probe runs are logged to `probes/probe_history.jsonl` in JSONL format:

```json
{
  "run_id": "2026-04-25-baseline",
  "timestamp": "2026-04-25T14:30:00Z",
  "results": {
    "P-001": {"triggered": true, "cost": 847, "notes": "..."},
    "P-002": {"triggered": false, "cost": 0, "notes": "..."}
  },
  "total_triggered": 12,
  "total_cost": 15847,
  "probe_count": 30
}
```

Each line is a complete JSON object representing one evaluation run.

## Interpreting Trends

### Sparkline Visualization

```
P-001: █░░█████░█  (7/10 triggered)
```

- `█` = Probe triggered in that run
- `░` = Probe passed in that run
- Shows last 10 runs, most recent on right

### Trend Icons

- 📈 **Worsening**: Probe triggered in >50% of last 3 runs
- 📉 **Improving**: Probe triggered in <50% of last 3 runs

### Priority Matrix

| Trigger Rate | Business Cost | Priority |
|--------------|---------------|----------|
| >70% | >$1000 | 🔴 Critical |
| >50% | >$500 | 🟠 High |
| >30% | >$200 | 🟡 Medium |
| <30% | <$200 | 🟢 Low |

Focus fixes on probes in the Critical and High categories.

## Example: Fixing a Regression

```bash
# Baseline run before changes
python probes/probe_monitor.py log \
  --run-id "2026-04-25-before-fix" \
  --results baseline_results.json

# Make code changes to fix P-003 (bench over-commitment)
# ...

# Run evaluation again
python eval/e2e_test.py

# Log new results
python probes/probe_monitor.py log \
  --run-id "2026-04-25-after-fix" \
  --results after_fix_results.json

# Check if P-003 is now passing
python probes/probe_monitor.py check

# Generate report to visualize improvement
python probes/probe_monitor.py report
```

## Maintenance

### Cleaning Old History

Keep last 50 runs to avoid unbounded growth:

```bash
tail -n 50 probes/probe_history.jsonl > probes/probe_history.jsonl.tmp
mv probes/probe_history.jsonl.tmp probes/probe_history.jsonl
```

### Adding New Probes

1. Add probe to `probes/probe_library.md`
2. Include probe ID in evaluation results JSON
3. Tool will automatically track new probe in future runs

### Archiving Historical Data

```bash
# Archive by quarter
mkdir -p probes/archive/2026-Q2
cp probes/probe_history.jsonl probes/archive/2026-Q2/
> probes/probe_history.jsonl  # Clear current log
```

## Troubleshooting

### "No probe history found"

You need to log at least one run first:

```bash
python probes/probe_monitor.py log \
  --run-id "initial-run" \
  --results probes/example_probe_results.json
```

### "Need at least 2 runs to detect regressions"

The regression check compares the last 2 runs. Log a second run:

```bash
python probes/probe_monitor.py log \
  --run-id "second-run" \
  --results new_results.json
```

### Probe not appearing in report

Ensure the probe ID matches exactly between:
- `probes/probe_library.md` (## P-XXX header)
- Results JSON (key name)

Probe IDs are case-sensitive.
