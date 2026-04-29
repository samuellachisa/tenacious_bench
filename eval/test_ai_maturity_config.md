# Test Description: AI Maturity Config-Driven Scoring

## Purpose

Verify that the AI maturity scorer correctly loads and uses configuration from `agent/config/ai_maturity_config.json`, enabling tuning without code changes.

## Functions Under Test

- `agent/core/enrichment.py::_load_ai_maturity_config()`
- `agent/core/enrichment.py::_get_default_ai_maturity_config()`
- `agent/core/enrichment.py::score_ai_maturity()` (config-driven version)

## Test Cases

### 1. Config File Loads Successfully

```python
from agent.core.enrichment import _load_ai_maturity_config

config = _load_ai_maturity_config()

assert "signals" in config
assert "confidence_rules" in config
assert "ai_adjacent_roles" in config["signals"]
assert "keywords" in config["signals"]["ai_adjacent_roles"]
assert "thresholds" in config["signals"]["ai_adjacent_roles"]
```

### 2. Fallback to Defaults if Config Missing

```python
import os
from pathlib import Path
from agent.core.enrichment import _load_ai_maturity_config

# Temporarily rename config file
config_path = Path("agent/config/ai_maturity_config.json")
backup_path = Path("agent/config/ai_maturity_config.json.bak")
config_path.rename(backup_path)

try:
    config = _load_ai_maturity_config()
    
    # Should return default config, not crash
    assert "signals" in config
    assert "ai_adjacent_roles" in config["signals"]
    
    # Should log trace event
    # Check Langfuse for "ai_maturity_config_load_failed" event
    
finally:
    # Restore config file
    backup_path.rename(config_path)
```

### 3. Scorer Uses Config Keywords

```python
from agent.core.enrichment import score_ai_maturity

# Test that scorer uses keywords from config
job_signals = {
    "ai_roles": ["ML Engineer", "Data Scientist", "AI Platform Lead"],
    "open_roles": 10
}
firmographics = {
    "description": "mlops platform for model deployment",
    "industry": "artificial intelligence",
    "recent_news": "CEO announces llm strategy",
    "cto_name": "Jane Doe",
    "cto_tenure_days": 30
}

result = score_ai_maturity(job_signals, firmographics)

# Should detect all signals based on config keywords
assert result["score"] == 3
assert result["confidence"] >= 0.85

# Check signal breakdown
signals = {s["signal_name"]: s for s in result["signal_breakdown"]}
assert signals["ai_adjacent_roles"]["detected"] == True
assert signals["named_ai_leadership"]["detected"] == True
assert signals["ai_industry_classification"]["detected"] == True
assert signals["executive_commentary"]["detected"] == True
assert signals["ml_stack_keywords"]["detected"] == True
```

### 4. Scorer Uses Config Thresholds

```python
from agent.core.enrichment import score_ai_maturity

# Test 3+ roles threshold (should give +2 score, 0.9 confidence)
job_signals = {"ai_roles": ["ML Engineer", "Data Scientist", "AI Lead"], "open_roles": 3}
firmographics = {"description": "", "industry": "", "recent_news": "", "cto_name": "", "cto_tenure_days": None}

result = score_ai_maturity(job_signals, firmographics)

# Config specifies: 3+ roles → score +2, confidence 0.9
assert result["score"] == 2
signals = {s["signal_name"]: s for s in result["signal_breakdown"]}
assert signals["ai_adjacent_roles"]["confidence"] == 0.9

# Test 1 role threshold (should give +1 score, 0.7 confidence)
job_signals = {"ai_roles": ["ML Engineer"], "open_roles": 1}
result = score_ai_maturity(job_signals, firmographics)

assert result["score"] == 1
signals = {s["signal_name"]: s for s in result["signal_breakdown"]}
assert signals["ai_adjacent_roles"]["confidence"] == 0.7
```

### 5. Scorer Uses Config Confidence Rules

```python
from agent.core.enrichment import score_ai_maturity

# Test high confidence rule: 2+ high-weight signals → 0.85 confidence
job_signals = {"ai_roles": ["ML Engineer", "Data Scientist", "AI Lead"], "open_roles": 3}
firmographics = {
    "description": "",
    "industry": "",
    "recent_news": "",
    "cto_name": "Jane Doe",
    "cto_tenure_days": 30
}

result = score_ai_maturity(job_signals, firmographics)

# 2 high-weight signals detected (ai_adjacent_roles + named_ai_leadership)
# Config specifies: 2+ high-weight → 0.85 confidence
assert result["confidence"] == 0.85

# Test medium confidence rule: 1 high-weight + 2 medium-weight → 0.70 confidence
job_signals = {"ai_roles": ["ML Engineer"], "open_roles": 1}
firmographics = {
    "description": "",
    "industry": "artificial intelligence",
    "recent_news": "CEO announces ai strategy",
    "cto_name": "",
    "cto_tenure_days": None
}

result = score_ai_maturity(job_signals, firmographics)

# 1 high-weight (ai_adjacent_roles) + 2 medium-weight (industry + commentary)
# Config specifies: 1 high + 2 medium → 0.70 confidence
assert result["confidence"] == 0.70
```

### 6. Config Changes Reflected Without Code Changes

**Setup**: Edit `agent/config/ai_maturity_config.json` to add "chatgpt" to executive_commentary keywords:

```json
{
  "signals": {
    "executive_commentary": {
      "keywords": ["ai", "machine learning", "llm", "automation", "chatgpt"]
    }
  }
}
```

**Test**:

```python
from agent.core.enrichment import score_ai_maturity

job_signals = {"ai_roles": [], "open_roles": 0}
firmographics = {
    "description": "",
    "industry": "",
    "recent_news": "Company launches chatgpt integration",
    "cto_name": "",
    "cto_tenure_days": None
}

result = score_ai_maturity(job_signals, firmographics)

# Should detect executive_commentary signal with new "chatgpt" keyword
signals = {s["signal_name"]: s for s in result["signal_breakdown"]}
assert signals["executive_commentary"]["detected"] == True
assert result["score"] >= 1
```

**Cleanup**: Revert config change after test.

### 7. Low-Weight Signals Don't Contribute to Score

```python
from agent.core.enrichment import score_ai_maturity

# Only low-weight signals present (ml_stack_keywords + strategic_ai_communications)
job_signals = {"ai_roles": [], "open_roles": 0}
firmographics = {
    "description": "ai-powered platform with mlops pipeline",
    "industry": "software",
    "recent_news": "",
    "cto_name": "",
    "cto_tenure_days": None
}

result = score_ai_maturity(job_signals, firmographics)

# Low-weight signals detected but score should be 0 (they only contribute to confidence)
assert result["score"] == 0
assert result["confidence"] < 0.60  # Low confidence (only low-weight signals)

signals = {s["signal_name"]: s for s in result["signal_breakdown"]}
assert signals["ml_stack_keywords"]["detected"] == True
assert signals["strategic_ai_communications"]["detected"] == True
```

### 8. Leadership Tenure Threshold Configurable

**Setup**: Edit config to change tenure threshold from 90 to 60 days:

```json
{
  "signals": {
    "named_ai_leadership": {
      "tenure_threshold_days": 60
    }
  }
}
```

**Test**:

```python
from agent.core.enrichment import score_ai_maturity

# CTO with 75 days tenure (between 60 and 90)
job_signals = {"ai_roles": [], "open_roles": 0}
firmographics = {
    "description": "",
    "industry": "",
    "recent_news": "",
    "cto_name": "Jane Doe",
    "cto_tenure_days": 75
}

result = score_ai_maturity(job_signals, firmographics)

# With threshold=60, CTO at 75 days should NOT trigger signal
signals = {s["signal_name"]: s for s in result["signal_breakdown"]}
assert signals["named_ai_leadership"]["detected"] == False

# With threshold=90 (default), CTO at 75 days SHOULD trigger signal
# (Revert config and re-test to verify)
```

**Cleanup**: Revert config to tenure_threshold_days: 90.

## Integration Test

```python
from agent.core.enrichment import run_enrichment_pipeline

# Full pipeline test with config-driven AI maturity scoring
result = await run_enrichment_pipeline("DataFlow Technologies")

ai_maturity = result.get("ai_maturity", {})

# Verify structure
assert "score" in ai_maturity
assert "confidence" in ai_maturity
assert "justification" in ai_maturity
assert "signal_breakdown" in ai_maturity

# Verify signal breakdown has all 6 signals
signal_names = {s["signal_name"] for s in ai_maturity["signal_breakdown"]}
expected_signals = {
    "ai_adjacent_roles",
    "named_ai_leadership",
    "ai_industry_classification",
    "executive_commentary",
    "ml_stack_keywords",
    "strategic_ai_communications"
}
assert signal_names == expected_signals

# Verify each signal has required fields
for signal in ai_maturity["signal_breakdown"]:
    assert "signal_name" in signal
    assert "weight" in signal
    assert "detected" in signal
    assert "confidence" in signal
    assert "evidence" in signal
```

## Benefits of Config-Driven Approach

1. **No code deployment for keyword changes**: Add "generative ai", "chatgpt", "claude" to keywords without touching Python code

2. **A/B testing**: Keep multiple config versions (config_v1.json, config_v2.json) and swap to compare results

3. **Rapid iteration**: Adjust thresholds (e.g., 3+ roles → 5+ roles for "very high" signal) and test immediately

4. **Domain-specific tuning**: Different configs for different ICPs (e.g., AI-native companies vs traditional enterprises)

5. **Audit trail**: Config changes tracked in git with commit messages explaining tuning rationale

6. **Non-technical tuning**: Sales/marketing can propose keyword additions without needing Python knowledge

## Monitoring

After config changes, monitor these Langfuse events:

- `ai_maturity_config_load_failed`: Config file missing or malformed (should be 0)
- `ai_maturity_score_distribution`: How many prospects score 0, 1, 2, 3 (should shift after tuning)
- `capability_gap_segment_rate`: % of prospects classified as capability_gap (requires score >= 2)

## Related Documentation

- **Tuning guide**: `docs/AI_MATURITY_TUNING.md`
- **Config file**: `agent/config/ai_maturity_config.json`
- **Implementation**: `agent/core/enrichment.py::score_ai_maturity()`
- **Probes**: `probes/probe_library.md` (P-023: false positive, P-024: false negative)
