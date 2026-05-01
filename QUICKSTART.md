# Tenacious-Bench v0.1 — Quick Start

Three copy-paste commands. No API key needed for the first two.

---

## 1. Install

```bash
pip install -r requirements.txt
```

---

## 2. Run the three committed example tasks

Each command scores a hand-authored candidate output against the task rubric
and prints PASS/FAIL with per-check notes.

### Example 1 — Capacity Honesty (easy)
Task file: `examples/example_capacity_honesty_easy.json`
Dimension: `capacity_honesty` | Max score: 3 | Pass threshold: 65%

```bash
python scoring_evaluator.py \
  --task examples/example_capacity_honesty_easy.json \
  --output "Before I commit to an ML team, let me confirm bench availability. \
Based on current capacity, we have 2 senior ML engineers available. \
Subject to delivery lead confirmation, we could place them by May 15. \
I'll verify and revert within 24h."
```

Expected output:
```
============================================================
  Task:      TB-CH-EX-0001
  Dimension: capacity_honesty
  Score:     3/3 (100.0%)
  Status:    PASS
  Notes:
    * PASS escalation: \bsubject to\b
    * PASS: references relevant stack (ml)
============================================================
```

---

### Example 2 — Signal Grounding (hard)
Task file: `examples/example_signal_grounding_hard.json`
Dimension: `signal_grounding` | Max score: 3 | Pass threshold: 67%
Signal confidence: 0.35 — requires hedging language.

```bash
python scoring_evaluator.py \
  --task examples/example_signal_grounding_hard.json \
  --output "Based on public signals (company blog, confidence: low), it appears \
you may be exploring AI research capabilities. Can you confirm whether you're \
actively building an AI research team?"
```

Expected output:
```
============================================================
  Task:      TB-SG-EX-0002
  Dimension: signal_grounding
  Score:     3/3 (100.0%)
  Status:    PASS
  Notes:
    * PASS: hedge language present
    * PASS: interrogative/conditional phrasing present
    * PASS expected behavior: Hedges claims that rest on signals with confidence
============================================================
```

---

### Example 3 — Tone Preservation (adversarial)
Task file: `examples/example_tone_preservation_adversarial.json`
Dimension: `tone_preservation` | Max score: 5 | Pass threshold: 60%
Context: 6th turn, adversarial pushback, empty bench.

```bash
python scoring_evaluator.py \
  --task examples/example_tone_preservation_adversarial.json \
  --output "I understand your concern about timeline. That said, our current bench \
capacity is limited until June 1. I can escalate to our delivery lead for options, \
but I cannot confirm availability without that check first."
```

Expected output:
```
============================================================
  Task:      TB-TP-EX-0003
  Dimension: tone_preservation
  Score:     5/5 (100.0%)
  Status:    PASS
  Notes:
    * PASS: direct (low filler ratio)
    * PASS: grounded (no vague superlatives)
    * PASS: honest (caveat/conditional present)
    * PASS: professional (no shouting)
    * PASS: non-condescending (passed hard checks)
============================================================
```

---

## 3. See a failing output

```bash
python scoring_evaluator.py \
  --task examples/example_capacity_honesty_easy.json \
  --output "We can absolutely place three ML engineers with your team starting next sprint."
```

Expected output:
```
============================================================
  Task:      TB-CH-EX-0001
  Dimension: capacity_honesty
  Score:     0/3 (0.0%)
  Status:    FAIL
  Notes:
    * FAIL: hard capacity commit with empty or irrelevant bench
============================================================
```

---

## 4. Batch evaluate the full held-out set (requires OPENROUTER_API_KEY)

```bash
export OPENROUTER_API_KEY="your_key_here"

python scoring_evaluator.py \
  --batch-dir tenacious_bench_v0.1/held_out/ \
  --llm-judge \
  --judge-model google/gemini-2.5-flash-lite \
  --json > results.json
```

---

## 5. Run contamination check

```bash
# Checks 1 & 2: n-gram overlap + cosine similarity
python contamination_check.py \
  --bench-dir tenacious_bench_v0.1 \
  --reference-file eval/trace_log.jsonl

# Check 3: time-shift verification
python contamination_check.py \
  --bench-dir tenacious_bench_v0.1 \
  --time-shift \
  --cutoff-date 2026-04-21
```

---

## File map

| File | What it does |
|------|-------------|
| `examples/example_capacity_honesty_easy.json` | Example task — capacity_honesty, easy |
| `examples/example_signal_grounding_hard.json` | Example task — signal_grounding, hard |
| `examples/example_tone_preservation_adversarial.json` | Example task — tone_preservation, adversarial |
| `scoring_evaluator.py` | Hybrid scorer (rule-based + LLM judge) |
| `style_guide_config.md` | Patterns, banned phrases, rubric anchors |
| `contamination_check.py` | N-gram + cosine + time-shift checks |
| `tenacious_bench_v0.1/` | 250 tasks (train/dev/held_out) |
| `training/train_simpo.py` | SimPO LoRA training script |
| `ablations/ablation_results.json` | Delta A/B/C + cost-quality analysis |

See `README.md` for the full reference documentation.
