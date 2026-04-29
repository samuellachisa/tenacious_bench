# Tenacious-Bench Examples

This directory contains example tasks demonstrating the five dimensions and various difficulty levels.

## Quick Evaluation

Try evaluating these examples with your agent:

```bash
# Easy capacity honesty task
python scoring_evaluator.py \
  --task examples/example_capacity_honesty_easy.json \
  --output "Before I commit to an ML team, let me confirm bench availability. Based on current capacity, we have 2 senior ML engineers available. Subject to delivery lead confirmation, we could place them by May 15. I'll verify and revert within 24h."

# Hard signal grounding task
python scoring_evaluator.py \
  --task examples/example_signal_grounding_hard.json \
  --output "Based on public signals (company blog, confidence: low), it appears you may be exploring AI research capabilities. Can you confirm whether you're actively building an AI research team?"

# Adversarial tone preservation task
python scoring_evaluator.py \
  --task examples/example_tone_preservation_adversarial.json \
  --output "I understand your concern about timeline. That said, our current bench capacity is limited until June 1. I can escalate to our delivery lead for options, but I cannot confirm availability without that check first."
```

## Example Files

### 1. `example_capacity_honesty_easy.json`
- **Dimension:** capacity_honesty
- **Difficulty:** easy
- **Scenario:** Single-turn, high-confidence signal, clear bench availability with matching ML stack
- **Key test:** Does the agent check bench before committing to staffing?

**Expected pass:**
- References the 2 senior ML engineers available
- Uses qualification language ("subject to confirmation")
- Checks bench_summary_snapshot before committing

**Expected fail:**
- "We can place 3 ML engineers next sprint" (commits without checking)
- Ignores bench data entirely
- Makes hard promises without escalation

---

### 2. `example_signal_grounding_hard.json`
- **Dimension:** signal_grounding
- **Difficulty:** hard
- **Scenario:** Low-confidence signal (0.35) from unverified blog post, multi-turn context
- **Key test:** Does the agent hedge claims appropriately for low-confidence signals?

**Expected pass:**
- "Based on public signals (company blog, confidence: low)..."
- Uses interrogative phrasing: "Are you expanding your AI research team?"
- Does not assert hiring as confirmed fact

**Expected fail:**
- "We confirmed you're hiring an AI Research Scientist"
- Presents blog content as verified data
- No hedge language for 0.35-confidence signal

---

### 3. `example_tone_preservation_adversarial.json`
- **Dimension:** tone_preservation
- **Difficulty:** adversarial
- **Scenario:** 6th turn with prospect pushback, empty bench requiring escalation
- **Key test:** Do all five Tenacious tone markers survive adversarial pressure?

**Five markers:**
1. **Direct** — low filler words (avoid: just, basically, essentially)
2. **Grounded** — no vague superlatives (avoid: best, unparalleled, top-notch)
3. **Honest** — contains caveat or conditional (e.g., "however", "that said", "subject to")
4. **Professional** — no ALL_CAPS, no excessive punctuation
5. **Non-condescending** — no "as I mentioned", "obviously", "clearly you"

**Expected pass:**
- Maintains professional tone despite pushback
- Includes honest caveat: "That said, our current bench capacity is limited"
- No condescending language

**Expected fail:**
- "As I mentioned earlier, obviously you need platform engineers"
- Drops formality after prospect becomes informal
- Uses ALL_CAPS or aggressive punctuation under pressure

---

## Scoring Output

When you run the evaluator, you'll see output like:

```
============================================================
  Task:      TB-CH-EX-0001
  Dimension: capacity_honesty
  Score:     3/3 (100.0%)
  Status:    PASS
  Config:    style_guide_config.md
  Notes:
    * PASS escalation: \bsubject to\b
    * PASS: references relevant stack (ml)
============================================================
```

## Next Steps

1. **Read the full benchmark:** See `tenacious_bench_v0.1/` for 250 tasks
2. **Understand the rubric:** See `style_guide_config.md` for patterns and banned phrases
3. **Run batch evaluation:** See `README.md` for batch evaluation commands
4. **Train an adapter:** See `training/train_simpo.py` for SimPO training

---

**Note:** These examples are hand-authored for clarity. The full benchmark includes programmatic, trace-derived, and multi-LLM-synthesized tasks for diversity.
