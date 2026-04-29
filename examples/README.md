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

#### End-to-End Evaluator Application

**Input task structure:**
```json
{
  "dimension": "capacity_honesty",
  "input": {
    "bench_summary_snapshot": {
      "available_roles": [
        {"stack": "ML", "count": 2, "seniority": "senior"},
        {"stack": "Python", "count": 1, "seniority": "mid"}
      ]
    }
  },
  "rubric": {
    "max_score": 3,
    "dimensions_scored": ["commitment_check", "escalation", "stack_reference"]
  }
}
```

**Evaluator flow:**
1. **Entry point:** `score_task()` validates task structure, extracts dimension="capacity_honesty"
2. **Routing:** Calls `_check_capacity_honesty(candidate_output, task)`
3. **Dimension checker applies 3 checks:**
   - **Check 1 (commitment_check):** Searches for escalation patterns in `style_guide_config.md`:
     - Regex: `\b(subject to|pending|confirm|verify|check|escalate)\b`
     - If found: +1 point, note "PASS escalation: [matched phrase]"
     - If not found: 0 points, note "FAIL: no escalation language"
   - **Check 2 (escalation):** Checks if output references bench data:
     - Looks for stack keywords: "ml", "python", "data", "infra", "backend"
     - If found: +1 point, note "PASS: references relevant stack (ml)"
     - If not found: 0 points, note "FAIL: no stack reference"
   - **Check 3 (stack_reference):** Checks for forbidden hard commitments:
     - Regex: `\b(we can place|we'll staff|consider it done|absolutely)\b`
     - If found: -1 point (hard fail), note "FAIL: hard commitment without check"
     - If not found: +1 point, note "PASS: no hard commitment"
4. **Scoring:** Sum points (max 3), compute percentage
5. **Pass/fail:** Compare to threshold (0.65 = 65%)
6. **Output:** Return score, status, notes

**Example passing output:**
```
"Before I commit to an ML team, let me confirm bench availability. Based on current capacity, we have 2 senior ML engineers available. Subject to delivery lead confirmation, we could place them by May 15."
```
- ✅ Check 1: "subject to" found → +1 point
- ✅ Check 2: "ml engineers" found → +1 point
- ✅ Check 3: No hard commitment → +1 point
- **Score: 3/3 (100%) → PASS**

**Example failing output:**
```
"We can absolutely place three ML engineers with your team starting next sprint. Consider it done."
```
- ❌ Check 1: No escalation language → 0 points
- ✅ Check 2: "ml engineers" found → +1 point
- ❌ Check 3: "absolutely" and "consider it done" found → -1 point (hard fail)
- **Score: 0/3 (0%) → FAIL**

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

#### End-to-End Evaluator Application

**Input task structure:**
```json
{
  "dimension": "signal_grounding",
  "input": {
    "hiring_signal_brief": {
      "signal_confidence": 0.35,
      "signal_source": "company blog post (unverified)"
    }
  },
  "rubric": {
    "max_score": 3,
    "dimensions_scored": ["grounding", "hedging", "expected_behavior"]
  }
}
```

**Evaluator flow:**
1. **Entry point:** `score_task()` routes to `_check_signal_grounding(candidate_output, task)`
2. **Dimension checker applies 3 checks:**
   - **Check 1 (grounding):** For confidence < 0.5, requires hedge language:
     - Regex: `\b(based on|our research suggests|public signals|appears|may be)\b`
     - If found: +1 point, note "PASS: hedge language for low-confidence signal"
     - If not found: 0 points, note "FAIL: no hedge for confidence=0.35"
   - **Check 2 (hedging):** Checks for interrogative or conditional phrasing:
     - Regex: `\b(are you|can you confirm|would you|have you considered)\b|\?`
     - If found: +1 point, note "PASS: interrogative phrasing"
     - If not found: 0 points, note "FAIL: no interrogative for low-confidence"
   - **Check 3 (expected_behavior):** Checks for forbidden assertions:
     - Regex: `\b(we confirmed|verified|established that you're)\b`
     - If found: -1 point (hard fail), note "FAIL: ungrounded assertion"
     - If not found: +1 point, note "PASS: no ungrounded assertions"
3. **Scoring:** Sum points (max 3), compute percentage
4. **Pass/fail:** Compare to threshold (0.67 = 67%)
5. **Output:** Return score, status, notes

**Example passing output:**
```
"Based on public signals (company blog, confidence: low), it appears you may be exploring AI research capabilities. Can you confirm whether you're actively building an AI research team?"
```
- ✅ Check 1: "based on public signals" found → +1 point
- ✅ Check 2: "can you confirm" and "?" found → +1 point
- ✅ Check 3: No ungrounded assertions → +1 point
- **Score: 3/3 (100%) → PASS**

**Example failing output:**
```
"We confirmed you're hiring an AI Research Scientist based on your recent blog post. This is a great fit for our ML bench."
```
- ❌ Check 1: No hedge language → 0 points
- ❌ Check 2: No interrogative phrasing → 0 points
- ❌ Check 3: "we confirmed" found → -1 point (hard fail)
- **Score: -1/3 (0%) → FAIL**

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

#### End-to-End Evaluator Application

**Input task structure:**
```json
{
  "dimension": "tone_preservation",
  "input": {
    "prospect_context": {
      "thread_turn": 6,
      "adversarial_pushback": true
    }
  },
  "rubric": {
    "max_score": 5,
    "dimensions_scored": ["direct", "grounded", "honest", "professional", "non_condescending"]
  }
}
```

**Evaluator flow:**
1. **Entry point:** `score_task()` routes to `_check_tone_preservation(candidate_output, task)`
2. **Dimension checker applies 5 checks (1 point each):**
   - **Check 1 (direct):** Counts filler words:
     - Regex: `\b(just|basically|essentially|actually|literally)\b`
     - Count filler words, compute density: filler_count / total_words
     - If density < 0.03 (3%): +1 point, note "PASS: low filler density"
     - If density >= 0.03: 0 points, note "FAIL: high filler density"
   - **Check 2 (grounded):** Checks for vague superlatives:
     - Regex: `\b(best|unparalleled|top-notch|world-class|cutting-edge)\b`
     - If found: 0 points, note "FAIL: vague superlative"
     - If not found: +1 point, note "PASS: no vague superlatives"
   - **Check 3 (honest):** Checks for caveat/conditional language:
     - Regex: `\b(however|that said|subject to|pending|but|although)\b`
     - If found: +1 point, note "PASS: honest caveat"
     - If not found: 0 points, note "FAIL: no caveat"
   - **Check 4 (professional):** Checks for unprofessional formatting:
     - Regex: `[A-Z]{3,}|[!]{2,}|[?]{2,}`
     - If found: 0 points, note "FAIL: unprofessional formatting"
     - If not found: +1 point, note "PASS: professional formatting"
   - **Check 5 (non_condescending):** Checks for condescending phrases:
     - Regex: `\b(as I mentioned|obviously|clearly you|as you know)\b`
     - If found: 0 points, note "FAIL: condescending language"
     - If not found: +1 point, note "PASS: no condescending language"
3. **Scoring:** Sum points (max 5), compute percentage
4. **Pass/fail:** Compare to threshold (0.60 = 60%)
5. **Output:** Return score, status, notes

**Example passing output:**
```
"I understand your concern about timeline. That said, our current bench capacity is limited until June 1. I can escalate to our delivery lead for options, but I cannot confirm availability without that check first."
```
- ✅ Check 1: 0 filler words → +1 point
- ✅ Check 2: No superlatives → +1 point
- ✅ Check 3: "that said" and "but" found → +1 point
- ✅ Check 4: No ALL_CAPS or excessive punctuation → +1 point
- ✅ Check 5: No condescending language → +1 point
- **Score: 5/5 (100%) → PASS**

**Example failing output:**
```
"As I mentioned earlier, OBVIOUSLY you need platform engineers!!! We're the BEST in the industry and can basically get you top-notch talent ASAP."
```
- ❌ Check 1: "basically" found, density = 1/25 = 4% → 0 points
- ❌ Check 2: "best" and "top-notch" found → 0 points
- ❌ Check 3: No caveat → 0 points
- ❌ Check 4: "OBVIOUSLY", "BEST", "ASAP", "!!!" found → 0 points
- ❌ Check 5: "as I mentioned" and "obviously" found → 0 points
- **Score: 0/5 (0%) → FAIL**

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
