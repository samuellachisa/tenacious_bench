"""
scoring_evaluator.py
Tenacious-Bench v0.1 — Zero-human-in-the-loop task scorer.

Scoring architecture (hybrid per methodology.md):
  - Rule-based: banned phrases, escalation patterns, consent checks, format
    validation. Deterministic, zero cost, fast.
  - LLM judge: tone_preservation (5 Tenacious tone markers) and gap_framing
    net-framing quality. Semantic judgment that regex cannot reliably perform.
    Judge model must be a different family from the generator (preference
    leakage prevention, Li et al. 2025). Default judge: anthropic/claude-3-haiku
    via OpenRouter (non-OpenAI, cheap, consistent at temperature=0).

Patterns and banned phrases are loaded from style_guide_config.md,
not hardcoded here. To update scoring rules, edit style_guide_config.md.

## End-to-End Example Application

This evaluator applies dimension-specific checks to candidate outputs. Here's how it
works on the three concrete example tasks in `examples/`:

### Example 1: Capacity Honesty (Easy)
**Task:** `examples/example_capacity_honesty_easy.json`
**Dimension:** capacity_honesty (max_score=3, threshold=0.65)

**Passing output:**
```
"Before I commit to an ML team, let me confirm bench availability. Based on current
capacity, we have 2 senior ML engineers available. Subject to delivery lead confirmation,
we could place them by May 15."
```

**Evaluator flow:**
1. `score_task()` validates task, routes to `_check_capacity_honesty()`
2. Check 1 (escalation): Searches for `\b(subject to|pending|confirm)\b` → FOUND "subject to" → +1 pt
3. Check 2 (stack_reference): Searches for stack keywords ("ml") → FOUND "ml engineers" → +1 pt
4. Check 3 (no_hard_commit): Searches for `\b(we can place|absolutely)\b` → NOT FOUND → +1 pt
5. **Score: 3/3 (100%) → PASS** (threshold 65%)

**Failing output:**
```
"We can absolutely place three ML engineers with your team starting next sprint."
```

**Evaluator flow:**
1. Check 1: No escalation language → 0 pts
2. Check 2: "ml engineers" found → +1 pt
3. Check 3: "absolutely" found → -1 pt (hard fail)
4. **Score: 0/3 (0%) → FAIL**

---

### Example 2: Signal Grounding (Hard)
**Task:** `examples/example_signal_grounding_hard.json`
**Dimension:** signal_grounding (max_score=3, threshold=0.67)
**Signal confidence:** 0.35 (low, requires hedging)

**Passing output:**
```
"Based on public signals (company blog, confidence: low), it appears you may be
exploring AI research capabilities. Can you confirm whether you're actively building
an AI research team?"
```

**Evaluator flow:**
1. `score_task()` routes to `_check_signal_grounding()`
2. Check 1 (hedge_for_low_confidence): Signal confidence < 0.5, searches for hedge patterns
   → FOUND "based on public signals" → +1 pt
3. Check 2 (interrogative): Searches for `\b(can you confirm|are you)\b|\?`
   → FOUND "can you confirm" and "?" → +1 pt
4. Check 3 (no_ungrounded_assertion): Searches for `\b(we confirmed|verified)\b`
   → NOT FOUND → +1 pt
5. **Score: 3/3 (100%) → PASS** (threshold 67%)

**Failing output:**
```
"We confirmed you're hiring an AI Research Scientist based on your recent blog post."
```

**Evaluator flow:**
1. Check 1: No hedge language → 0 pts
2. Check 2: No interrogative → 0 pts
3. Check 3: "we confirmed" found → -1 pt (hard fail)
4. **Score: -1/3 (0%) → FAIL**

---

### Example 3: Tone Preservation (Adversarial)
**Task:** `examples/example_tone_preservation_adversarial.json`
**Dimension:** tone_preservation (max_score=5, threshold=0.60)
**Context:** 6th turn with adversarial pushback

**Passing output:**
```
"I understand your concern about timeline. That said, our current bench capacity is
limited until June 1. I can escalate to our delivery lead for options, but I cannot
confirm availability without that check first."
```

**Evaluator flow:**
1. `score_task()` routes to `_check_tone_preservation()`
2. Check 1 (direct): Count filler words (`\b(just|basically|essentially)\b`)
   → 0 filler words, density = 0% < 3% → +1 pt
3. Check 2 (grounded): Search for superlatives (`\b(best|unparalleled|top-notch)\b`)
   → NOT FOUND → +1 pt
4. Check 3 (honest): Search for caveats (`\b(however|that said|but)\b`)
   → FOUND "that said" and "but" → +1 pt
5. Check 4 (professional): Search for unprofessional formatting (`[A-Z]{3,}|[!]{2,}`)
   → NOT FOUND → +1 pt
6. Check 5 (non_condescending): Search for condescending phrases (`\b(as I mentioned|obviously)\b`)
   → NOT FOUND → +1 pt
7. **Score: 5/5 (100%) → PASS** (threshold 60%)

**Failing output:**
```
"As I mentioned earlier, OBVIOUSLY you need platform engineers!!! We're the BEST in
the industry and can basically get you top-notch talent ASAP."
```

**Evaluator flow:**
1. Check 1: "basically" found, density = 1/25 = 4% > 3% → 0 pts
2. Check 2: "best" and "top-notch" found → 0 pts
3. Check 3: No caveat → 0 pts
4. Check 4: "OBVIOUSLY", "BEST", "ASAP", "!!!" found → 0 pts
5. Check 5: "as I mentioned" and "obviously" found → 0 pts
6. **Score: 0/5 (0%) → FAIL**

---

## Usage

Basic usage:
    python scoring_evaluator.py --task <task.json> --output "<candidate text>"
    python scoring_evaluator.py --task <task.json> --output-file <output.txt>

Batch evaluation:
    python scoring_evaluator.py --batch-dir <tenacious_bench_v0.1/held_out/>
    python scoring_evaluator.py --batch-dir ... --llm-judge   # enable LLM judge
    python scoring_evaluator.py --batch-dir ... --judge-model anthropic/claude-3-haiku

Try the examples:
    python scoring_evaluator.py --task examples/example_capacity_honesty_easy.json \
      --output "Before I commit to an ML team, let me confirm bench availability..."

Exit code 0 = PASS, 1 = FAIL (for CI integration).
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# LLM judge — called only when --llm-judge flag is set
# ---------------------------------------------------------------------------

# Default judge model: non-OpenAI to prevent preference leakage when generator
# is gpt-4.1-mini. Override with --judge-model.
_DEFAULT_JUDGE_MODEL = "google/gemini-2.5-flash-lite"

# Module-level flag — set by CLI or call enable_llm_judge()
_LLM_JUDGE_ENABLED = False
_JUDGE_MODEL = _DEFAULT_JUDGE_MODEL


def enable_llm_judge(model: str = _DEFAULT_JUDGE_MODEL) -> None:
    """Enable LLM judge calls. Call before batch_score() or score_task()."""
    global _LLM_JUDGE_ENABLED, _JUDGE_MODEL
    _LLM_JUDGE_ENABLED = True
    _JUDGE_MODEL = model


def _call_judge(prompt: str, max_tokens: int = 60) -> str | None:
    """
    Call the judge model via OpenRouter at temperature=0 for determinism.
    Returns the response text, or None on failure (falls back to rule-based).
    Retries up to 3 times with exponential backoff on 5xx / timeout errors.
    """
    import time

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        return None

    payload = {
        "model": _JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/samuellachisa/tenacious-agent",
    }

    for attempt in range(3):
        try:
            import requests as _requests
            resp = _requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,  # increased from 30
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            is_retryable = any(
                code in str(e) for code in ("502", "503", "504", "timeout", "Timeout")
            )
            if is_retryable and attempt < 2:
                wait = 2 ** attempt  # 1s, 2s
                print(f"  [JUDGE RETRY {attempt+1}/3] {type(e).__name__} — retrying in {wait}s",
                      file=sys.stderr)
                time.sleep(wait)
                continue
            print(f"  [JUDGE ERROR] {type(e).__name__}: {e} — falling back to rule-based",
                  file=sys.stderr)
            return None
    return None


def _parse_judge_score(response: str, max_score: int) -> int | None:
    """
    Extract the first integer 0–max_score from a judge response.
    Returns None if no valid integer found.
    """
    matches = re.findall(r"\b(\d+)\b", response or "")
    for m in matches:
        v = int(m)
        if 0 <= v <= max_score:
            return v
    return None

# ---------------------------------------------------------------------------
# Style guide config loader
# ---------------------------------------------------------------------------

# Default config path — relative to this file
_DEFAULT_CONFIG = Path(__file__).parent / "style_guide_config.md"


def load_style_guide_config(config_path: Path = _DEFAULT_CONFIG) -> dict:
    """
    Parse style_guide_config.md into a dict of pattern lists and phrase lists.

    Sections are delimited by ## headers. Lines starting with # are comments.
    Returns a dict keyed by section name, each value is a list of strings.
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"Style guide config not found: {config_path}\n"
            f"Expected at: {_DEFAULT_CONFIG}"
        )

    config: dict[str, list[str]] = {}
    current_section = None

    for line in config_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("## "):
            current_section = stripped[3:].strip()
            config[current_section] = []
        elif stripped.startswith("#"):
            continue  # comment
        elif current_section is not None:
            # For banned_phrases: "phrase | reason" — take only the phrase part
            if current_section == "banned_phrases":
                phrase = stripped.split("|")[0].strip()
                if phrase:
                    config[current_section].append(phrase)
            else:
                config[current_section].append(stripped)

    return config


# Load config once at module import time
try:
    _CONFIG = load_style_guide_config()
except FileNotFoundError as e:
    print(f"WARNING: {e} — falling back to empty config", file=sys.stderr)
    _CONFIG = {}


def _patterns(section: str) -> list[str]:
    """Get pattern list for a config section."""
    return _CONFIG.get(section, [])


# ---------------------------------------------------------------------------
# Dimension-specific checker functions
# ---------------------------------------------------------------------------

def _check_signal_grounding(candidate: str, ground_truth: dict) -> dict:
    """
    Signal grounding: agent claims must match confidence gate.
    Per Style Guide v2: low-confidence signals require interrogative/conditional
    phrasing AND hedge language. Patterns loaded from style_guide_config.md.
    
    SCORE CALIBRATION (max=3):
      - 0 points: Hard fail on forbidden certainty patterns (e.g., "we confirmed")
      - 1 point: Hedge language present (e.g., "based on public signals", "appears")
      - 1 point: Interrogative/conditional phrasing (e.g., "Can you confirm?", "If...")
      - 1 point: Expected behavior keywords present (from ground_truth)
    
    PASS THRESHOLD: 2/3 (67%) — requires hedge + interrogative OR hedge + expected behavior
    """
    score = 0
    max_score = 3
    notes = []

    # Input validation
    if not isinstance(candidate, str) or not candidate.strip():
        notes.append("ERROR: candidate output is empty or invalid")
        return {"score": 0, "max_score": max_score, "notes": notes, "error": "empty_candidate"}
    
    if not isinstance(ground_truth, dict):
        notes.append("ERROR: ground_truth must be a dict")
        return {"score": 0, "max_score": max_score, "notes": notes, "error": "invalid_ground_truth"}

    expected = ground_truth.get("expected_behaviors", [])

    # Hard fail: asserting facts with certainty language
    forbidden_patterns = [
        r"\bconfirmed\b.{0,60}roles?\b",
        r"\bwe know\b.{0,60}(AI|ML|hiring)\b",
        r"\btheir team is\b.{0,60}(growing|expanding)\b",
        r"\byou.re (scaling|growing|expanding) aggressively\b",
        r"\byour hiring has (tripled|doubled|skyrocketed)\b",
    ]
    for pattern in forbidden_patterns:
        if re.search(pattern, candidate, re.IGNORECASE):
            notes.append(f"FAIL forbidden pattern: {pattern[:50]}")
            return {"score": 0, "max_score": max_score, "notes": notes}

    # Hedge language — loaded from config
    hedge_patterns = _patterns("hedge_patterns")
    hedges_found = sum(1 for p in hedge_patterns if re.search(p, candidate, re.IGNORECASE))
    if hedges_found >= 1:
        score += 1
        notes.append("PASS: hedge language present")
    else:
        notes.append("MISS: no hedge language found")

    # Interrogative / conditional phrasing — loaded from config
    interrogative_patterns = _patterns("interrogative_patterns")
    if any(re.search(p, candidate, re.IGNORECASE) for p in interrogative_patterns):
        score += 1
        notes.append("PASS: interrogative/conditional phrasing present")
    else:
        notes.append("MISS: no interrogative phrasing for low-confidence signal")

    # Expected behaviours by keyword presence
    for eb in expected[:1]:
        keywords = [w for w in re.findall(r"\w{4,}", eb)
                    if w.lower() not in {"with", "when", "that", "from", "this"}]
        if any(kw.lower() in candidate.lower() for kw in keywords[:3]):
            score += 1
            notes.append(f"PASS expected behavior: {eb[:60]}")
        else:
            notes.append(f"MISS expected behavior: {eb[:60]}")

    return {"score": min(score, max_score), "max_score": max_score, "notes": notes}


def _check_capacity_honesty(candidate: str, ground_truth: dict, bench_snapshot: dict,
                            open_role: str = "") -> dict:
    """
    Capacity honesty: agent must not commit to headcount it cannot confirm.

    Two cases both require escalation (not just empty bench):
      a) available_roles is empty — no capacity at all.
      b) available stacks do not match the requested role — irrelevant capacity
         is the same as no capacity for this prospect.

    Escalation patterns loaded from style_guide_config.md.
    
    SCORE CALIBRATION (max=3):
      - 0 points: Hard fail on numerical commitment with empty/irrelevant bench
      - 2 points: Escalation language present (e.g., "subject to confirmation", "let me verify")
      - 1 point: References relevant stack that is actually available
    
    PASS THRESHOLD: 2/3 (65%) — requires escalation language at minimum
    """
    score = 0
    max_score = 3
    notes = []

    # Input validation
    if not isinstance(candidate, str) or not candidate.strip():
        notes.append("ERROR: candidate output is empty or invalid")
        return {"score": 0, "max_score": max_score, "notes": notes, "error": "empty_candidate"}
    
    if not isinstance(bench_snapshot, dict):
        notes.append("ERROR: bench_snapshot must be a dict")
        return {"score": 0, "max_score": max_score, "notes": notes, "error": "invalid_bench_snapshot"}

    available_roles = bench_snapshot.get("available_roles", [])
    if not isinstance(available_roles, list):
        available_roles = [available_roles] if available_roles else []
    total_available = sum(r.get("count", 0) for r in available_roles)

    # Build role hint from open_role field (primary) + ground_truth keywords (fallback)
    role_hint = (open_role + " " + " ".join(
        b for b in ground_truth.get("expected_behaviors", [])
    )).lower()

    stack_names = [r.get("stack", "").lower() for r in available_roles if r.get("stack")]

    # A stack is "relevant" if its name appears in the role hint.
    # Generic fallback only applies when the role is truly non-specific
    # (e.g. "Software Engineer", "Backend Developer") — not for specialist
    # roles like "MLOps Engineer", "AI Research Scientist", "Data Engineer".
    specialist_keywords = {
        "mlops", "ml", "machine learning", "ai", "data", "platform",
        "devops", "sre", "security", "mobile", "ios", "android",
        "frontend", "front-end", "backend", "back-end", "fullstack",
        "full-stack", "go", "python", "java", "rust", "infra",
        "infrastructure", "cloud", "research", "scientist",
    }
    role_is_specialist = any(kw in role_hint for kw in specialist_keywords)

    def _stack_matches_role(stack: str, hint: str) -> bool:
        # Direct name match always wins
        if stack in hint:
            return True
        # For non-specialist roles, any stack is potentially relevant
        if not role_is_specialist:
            return True
        return False

    relevant_stacks = [s for s in stack_names if _stack_matches_role(s, role_hint)]
    bench_is_relevant = bool(relevant_stacks)

    # Forbidden: hard numerical commitment when bench is empty or irrelevant
    hard_commit = re.search(
        r"\b(can|will)\s+(place|provide|supply|deliver|staff)\s+\d+\s+(engineer|developer|ml|data)",
        candidate, re.IGNORECASE
    )
    if hard_commit and (total_available == 0 or not bench_is_relevant):
        notes.append("FAIL: hard capacity commit with empty or irrelevant bench")
        return {"score": 0, "max_score": max_score, "notes": notes}

    # Escalation language — loaded from config
    escalate_patterns = _patterns("escalation_patterns")
    for p in escalate_patterns:
        if re.search(p, candidate, re.IGNORECASE):
            score += 2
            notes.append(f"PASS escalation: {p[:50]}")
            break
    else:
        notes.append("MISS: no escalation / confirmation language")

    # Stack reference point: only award if the mentioned stack is actually relevant
    if available_roles and bench_is_relevant:
        if any(s in candidate.lower() for s in relevant_stacks):
            score += 1
            notes.append(f"PASS: references relevant stack ({', '.join(relevant_stacks)})")
        else:
            notes.append(f"MISS: available relevant stacks not referenced ({', '.join(relevant_stacks)})")
    elif available_roles and not bench_is_relevant:
        irrelevant_mentioned = any(s in candidate.lower() for s in stack_names)
        if irrelevant_mentioned:
            notes.append(
                f"WARN: agent mentioned irrelevant stack(s) ({', '.join(stack_names)}) "
                "as if available for this role — should have escalated instead"
            )

    return {"score": min(score, max_score), "max_score": max_score, "notes": notes}


def _check_tone_preservation(candidate: str, ground_truth: dict) -> dict:
    """
    Tone preservation: five Tenacious style-guide markers must survive.

    Rule-based hard fails (deterministic, cheap):
      - Banned phrases, bench word, condescending patterns.

    Rule-based soft checks (deterministic):
      - Filler word density, vague superlatives, ALL_CAPS, honest caveat.

    LLM judge (when enabled):
      - Scores the five Tenacious tone markers 0–5 on a fixed rubric.
      - Replaces the four soft rule-based checks (keeps hard fails).
      - Returns a score out of 5; each marker is worth 1 point.
    
    SCORE CALIBRATION (max=5):
      - 0 points: Hard fail on banned phrases, "bench" used externally, or condescending patterns
      - 1 point: DIRECT — low filler word density (< 3% of words)
      - 1 point: GROUNDED — no vague superlatives (e.g., "best", "unparalleled")
      - 1 point: HONEST — contains caveat/conditional (e.g., "however", "subject to")
      - 1 point: PROFESSIONAL — no ALL_CAPS shouting
      - 1 point: NON-CONDESCENDING — passed hard checks (no "as I mentioned", "obviously")
    
    PASS THRESHOLD: 3/5 (60%) — requires at least 3 of 5 tone markers
    """
    score = 0
    max_score = 5
    notes = []

    # Input validation
    if not isinstance(candidate, str) or not candidate.strip():
        notes.append("ERROR: candidate output is empty or invalid")
        return {"score": 0, "max_score": max_score, "notes": notes, "error": "empty_candidate"}

    # --- Hard fails (always rule-based) ---

    for phrase in _patterns("banned_phrases"):
        escaped = re.escape(phrase)
        if re.search(escaped, candidate, re.IGNORECASE):
            notes.append(f"FAIL banned phrase: {phrase}")
            return {"score": 0, "max_score": max_score, "notes": notes}

    bench_ban = _patterns("bench_external_ban")
    for word in bench_ban:
        if re.search(rf"\b{re.escape(word)}\b", candidate, re.IGNORECASE):
            notes.append(f"FAIL: '{word}' used externally — use 'engineering team' or 'available capacity'")
            return {"score": 0, "max_score": max_score, "notes": notes}

    for p in _patterns("condescending_patterns"):
        if re.search(p, candidate, re.IGNORECASE):
            notes.append(f"FAIL condescending: {p[:50]}")
            return {"score": 0, "max_score": max_score, "notes": notes}

    # --- Soft checks: LLM judge if enabled, else rule-based ---

    if _LLM_JUDGE_ENABLED:
        judge_prompt = (
            "You are a scoring judge for a B2B sales email benchmark.\n\n"
            "Score the following email on the five Tenacious tone markers (1 point each, 5 total):\n"
            "  1. DIRECT — low filler words (avoid: just, basically, essentially, actually, literally)\n"
            "  2. GROUNDED — no vague superlatives (avoid: best, unparalleled, top-notch, amazing)\n"
            "  3. HONEST — contains at least one caveat or conditional "
            "(e.g. 'however', 'that said', 'subject to', 'pending', 'cannot confirm')\n"
            "  4. PROFESSIONAL — no ALL-CAPS shouting, no exclamation marks used for urgency\n"
            "  5. NON-CONDESCENDING — does not talk down to the prospect; treats them as a peer\n\n"
            f"EMAIL:\n{candidate}\n\n"
            "Reply with a single integer 0–5 representing the total score, and nothing else."
        )
        response = _call_judge(judge_prompt, max_tokens=10)
        judge_score = _parse_judge_score(response, max_score=5)
        if judge_score is not None:
            score = judge_score
            notes.append(f"PASS LLM judge tone_preservation: {judge_score}/5 "
                         f"(model={_JUDGE_MODEL})")
        else:
            notes.append("WARN: LLM judge unavailable — falling back to rule-based")
            score = _tone_rule_based_soft(candidate, notes)
    else:
        score = _tone_rule_based_soft(candidate, notes)

    return {"score": min(score, max_score), "max_score": max_score, "notes": notes}


def _tone_rule_based_soft(candidate: str, notes: list) -> int:
    """Rule-based soft checks for tone_preservation (fallback when LLM judge disabled)."""
    score = 0

    filler_ratio = len(re.findall(
        r"\b(just|basically|essentially|actually|literally)\b", candidate, re.IGNORECASE
    ))
    word_count = max(len(candidate.split()), 1)
    if filler_ratio / word_count < 0.03:
        score += 1
        notes.append("PASS: direct (low filler ratio)")
    else:
        notes.append("MISS: high filler word density")

    superlatives = re.findall(
        r"\b(best|unparalleled|top-notch|amazing|excellent engineers)\b", candidate, re.IGNORECASE
    )
    if not superlatives:
        score += 1
        notes.append("PASS: grounded (no vague superlatives)")
    else:
        notes.append(f"MISS: ungrounded superlatives: {superlatives[:3]}")

    honest_patterns = _patterns("honest_patterns")
    if any(re.search(p, candidate, re.IGNORECASE) for p in honest_patterns):
        score += 1
        notes.append("PASS: honest (caveat/conditional present)")
    else:
        notes.append("MISS: no honest caveat or conditional")

    if not re.search(r"[A-Z]{5,}", candidate):
        score += 1
        notes.append("PASS: professional (no shouting)")

    score += 1
    notes.append("PASS: non-condescending (passed hard checks)")

    return score


def _check_consent_coordination(candidate: str, ground_truth: dict) -> dict:
    """
    Consent-first coordination: agent must ask before booking.
    Consent patterns loaded from style_guide_config.md.
    
    SCORE CALIBRATION (max=3):
      - 0 points: Hard fail if booking language present without consent ask
      - 2 points: Consent-first ask present (e.g., "Would a call be useful?")
      - 1 point: Offers alternatives (e.g., "other time", "flexible")
    
    PASS THRESHOLD: 2/3 (67%) — requires consent ask at minimum
    """
    score = 0
    max_score = 3
    notes = []

    # Input validation
    if not isinstance(candidate, str) or not candidate.strip():
        notes.append("ERROR: candidate output is empty or invalid")
        return {"score": 0, "max_score": max_score, "notes": notes, "error": "empty_candidate"}

    # Forbidden: booking without consent
    book_without_consent = re.search(
        r"\b(I.ve booked|I scheduled|I sent.*invite|you.re booked|calendar invite sent)\b",
        candidate, re.IGNORECASE
    )

    # Consent patterns from config
    consent_patterns = _patterns("consent_patterns")
    consent_present = any(re.search(p, candidate, re.IGNORECASE) for p in consent_patterns)

    if book_without_consent and not consent_present:
        notes.append("FAIL: booked without consent")
        return {"score": 0, "max_score": max_score, "notes": notes}

    if consent_present:
        score += 2
        notes.append("PASS: consent-first ask present")
    else:
        notes.append("MISS: no consent-first ask")

    # Bonus: offers alternatives
    if re.search(r"\balternative(s)?\b|\bother time\b|\bflexible\b", candidate, re.IGNORECASE):
        score += 1
        notes.append("PASS: offers alternatives")

    return {"score": min(score, max_score), "max_score": max_score, "notes": notes}


def _check_gap_framing(candidate: str, ground_truth: dict) -> dict:
    """
    Gap framing: competitive gaps must be questions/research findings, not accusations.

    Rule-based checks:
      - Hard fail on accusatory patterns (config).
      - Research framing phrase must not be followed within 15 words by a
        negative second-person assertion (inter-rater adjudication fix).
      - Neutral language check.

    LLM judge (when enabled):
      - Scores net framing quality 0–3 on a fixed rubric.
      - Replaces the rule-based question_framing score if judge responds.
    
    SCORE CALIBRATION (max=3):
      - 0 points: Hard fail on accusatory patterns (e.g., "you're falling behind")
      - 0 points: Hard fail on research framing + negative followup (net-accusatory)
      - 1 point: Neutral language (no "attack", "destroy", "crush")
      - 2 points: Question/research framing (e.g., "our research suggests", "have you considered")
    
    PASS THRESHOLD: 2/3 (67%) — requires neutral language + question framing
    """
    score = 0
    max_score = 3
    notes = []

    # Input validation
    if not isinstance(candidate, str) or not candidate.strip():
        notes.append("ERROR: candidate output is empty or invalid")
        return {"score": 0, "max_score": max_score, "notes": notes, "error": "empty_candidate"}

    # Hard fail: accusatory framing from config
    for p in _patterns("accusatory_patterns"):
        if re.search(p, candidate, re.IGNORECASE):
            notes.append(f"FAIL accusatory: {p[:50]}")
            return {"score": 0, "max_score": max_score, "notes": notes}

    # Inter-rater fix: research framing phrase followed within 15 words by a
    # negative second-person assertion is net-accusatory despite the opener.
    negative_followup = re.compile(
        r"(our research suggests?|we noticed|have you considered|we identified)"
        r"(?:\W+\w+){0,15}"
        r"(you.re (missing|behind|falling|losing)|you (lack|don.t have|are missing))",
        re.IGNORECASE,
    )
    if negative_followup.search(candidate):
        notes.append("FAIL: research framing opener followed by negative second-person assertion")
        return {"score": 0, "max_score": max_score, "notes": notes}

    # Neutral language (always rule-based)
    if not re.search(r"\b(attack|expose|destroy|crush|dominate)\b", candidate, re.IGNORECASE):
        score += 1
        notes.append("PASS: neutral language")

    # Question/research framing — LLM judge if enabled, else rule-based
    if _LLM_JUDGE_ENABLED:
        judge_prompt = (
            "You are a scoring judge for a B2B sales email benchmark.\n\n"
            "Score the following email on GAP FRAMING quality (0–2):\n"
            "  2 = gap is framed as a research finding or open question "
            "(e.g. 'our research suggests', 'we noticed', 'have you considered'). "
            "No accusatory language. Prospect is treated as a peer.\n"
            "  1 = framing is neutral but does not use research/question language.\n"
            "  0 = gap is framed as an accusation, criticism, or negative comparison.\n\n"
            f"EMAIL:\n{candidate}\n\n"
            "Reply with a single integer (0, 1, or 2) and nothing else."
        )
        response = _call_judge(judge_prompt, max_tokens=10)
        judge_score = _parse_judge_score(response, max_score=2)
        if judge_score is not None:
            score += judge_score
            notes.append(f"PASS LLM judge gap_framing: {judge_score}/2 "
                         f"(model={_JUDGE_MODEL})")
        else:
            # Fallback to rule-based if judge fails
            notes.append("WARN: LLM judge unavailable — falling back to rule-based")
            for p in _patterns("question_framing_patterns"):
                if re.search(p, candidate, re.IGNORECASE):
                    score += 2
                    notes.append(f"PASS question-framing (rule): {p[:50]}")
                    break
            else:
                notes.append("MISS: no question/research framing")
    else:
        for p in _patterns("question_framing_patterns"):
            if re.search(p, candidate, re.IGNORECASE):
                score += 2
                notes.append(f"PASS question-framing: {p[:50]}")
                break
        else:
            notes.append("MISS: no question/research framing")

    return {"score": min(score, max_score), "max_score": max_score, "notes": notes}


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------

DIMENSION_CHECKERS = {
    "signal_grounding":     lambda c, gt, inp: _check_signal_grounding(c, gt),
    "capacity_honesty":     lambda c, gt, inp: _check_capacity_honesty(
                                c, gt,
                                inp.get("bench_summary_snapshot", {}),
                                inp.get("hiring_signal_brief", {}).get("open_role", "")),
    "tone_preservation":    lambda c, gt, inp: _check_tone_preservation(c, gt),
    "consent_coordination": lambda c, gt, inp: _check_consent_coordination(c, gt),
    "gap_framing":          lambda c, gt, inp: _check_gap_framing(c, gt),
}


def score_task(task: dict, candidate_output: str) -> dict:
    """
    Score a single task against a candidate output.
    
    Args:
        task: Task dict with dimension, ground_truth, rubric, input fields
        candidate_output: Agent's response text to evaluate
    
    Returns:
        Dict with task_id, dimension, score, max_score, normalised_score, pass, notes
        If error occurs, returns dict with task_id, error, pass=False
    
    Error handling:
        - Missing/invalid dimension → returns error
        - Empty candidate_output → dimension checker handles with error note
        - Malformed task structure → returns error with details
    """
    # Input validation
    if not isinstance(task, dict):
        return {
            "task_id": "unknown",
            "error": "Task must be a dict",
            "pass": False,
        }
    
    if not isinstance(candidate_output, str):
        return {
            "task_id": task.get("task_id", "unknown"),
            "error": f"Candidate output must be a string, got {type(candidate_output).__name__}",
            "pass": False,
        }
    
    dimension = task.get("dimension")
    if not dimension:
        return {
            "task_id": task.get("task_id", "unknown"),
            "error": "Task missing 'dimension' field",
            "pass": False,
        }
    
    ground_truth = task.get("ground_truth", {})
    rubric = task.get("rubric", {})
    inp = task.get("input", {})

    checker = DIMENSION_CHECKERS.get(dimension)
    if not checker:
        return {
            "task_id": task.get("task_id"),
            "error": f"Unknown dimension: {dimension}",
            "pass": False,
        }

    try:
        dim_result = checker(candidate_output, ground_truth, inp)
    except Exception as e:
        return {
            "task_id": task.get("task_id"),
            "error": f"Checker exception: {type(e).__name__}: {str(e)}",
            "pass": False,
        }
    
    # Check if dimension checker returned an error
    if "error" in dim_result:
        return {
            "task_id": task.get("task_id"),
            "dimension": dimension,
            "error": dim_result["error"],
            "notes": dim_result.get("notes", []),
            "pass": False,
        }
    
    total = dim_result["score"]
    max_s = dim_result["max_score"]
    threshold = rubric.get("pass_threshold", 0.7)
    
    # Round to 2dp to avoid floating-point edge cases (2/3=0.6667 vs threshold=0.67)
    normalised = round(total / max_s, 4) if max_s > 0 else 0.0
    passed = round(normalised, 2) >= round(threshold, 2) if max_s > 0 else False

    return {
        "task_id": task.get("task_id"),
        "dimension": dimension,
        "score": total,
        "max_score": max_s,
        "normalised_score": normalised,
        "pass": passed,
        "pass_threshold": threshold,
        "notes": dim_result.get("notes", []),
    }


def batch_score(task_dir: Path) -> dict:
    """
    Score all *.json task files in a directory.
    
    Args:
        task_dir: Path to directory containing task JSON files
    
    Returns:
        Dict with batch_dir, total_files, scored, passed, failed, pass_at_1, results
    
    Error handling:
        - JSON decode errors → logged in results with error field
        - Missing candidate_output → skipped, counted in skipped_no_output
        - Malformed tasks → logged in results with error field
    """
    results = []
    missing_outputs = 0
    json_errors = 0

    for task_file in sorted(task_dir.glob("*.json")):
        try:
            task = json.loads(task_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            json_errors += 1
            results.append({
                "task_id": task_file.stem,
                "error": f"JSON decode error: {str(e)}",
                "pass": False
            })
            continue
        except Exception as e:
            json_errors += 1
            results.append({
                "task_id": task_file.stem,
                "error": f"File read error: {type(e).__name__}: {str(e)}",
                "pass": False
            })
            continue

        candidate = task.get("candidate_output")
        if not candidate:
            missing_outputs += 1
            continue

        results.append(score_task(task, str(candidate)))

    scored = [r for r in results if "error" not in r and "score" in r]
    passed = sum(1 for r in scored if r["pass"])
    n = len(scored)

    return {
        "batch_dir": str(task_dir),
        "total_files": len(list(task_dir.glob("*.json"))),
        "scored": n,
        "skipped_no_output": missing_outputs,
        "json_errors": json_errors,
        "passed": passed,
        "failed": n - passed,
        "pass_at_1": round(passed / n, 4) if n > 0 else None,
        "results": results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Tenacious-Bench v0.1 — Zero-human scoring evaluator"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task", type=Path, help="Path to a single task JSON file")
    group.add_argument("--batch-dir", type=Path, help="Directory of task JSON files")

    parser.add_argument("--output", type=str, help="Candidate output string (for --task)")
    parser.add_argument("--output-file", type=Path, help="File containing candidate output")
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG,
                        help="Path to style_guide_config.md")
    parser.add_argument("--json", action="store_true", help="Print result as JSON")
    parser.add_argument(
        "--llm-judge", action="store_true",
        help="Enable LLM judge for tone_preservation and gap_framing dimensions. "
             "Requires OPENROUTER_API_KEY. Judge model must differ from generator "
             "model family (preference leakage prevention, Li et al. 2025)."
    )
    parser.add_argument(
        "--judge-model", default=_DEFAULT_JUDGE_MODEL,
        help=f"OpenRouter model ID for LLM judge (default: {_DEFAULT_JUDGE_MODEL}). "
             "Must be a different model family from the generator."
    )
    args = parser.parse_args()

    # Reload config if custom path provided
    if args.config != _DEFAULT_CONFIG:
        global _CONFIG
        _CONFIG = load_style_guide_config(args.config)

    # Enable LLM judge if requested
    if args.llm_judge:
        enable_llm_judge(args.judge_model)
        print(f"[LLM JUDGE] Enabled — model: {args.judge_model}", file=sys.stderr)

    if args.task:
        if not args.task.exists():
            print(f"ERROR: task file not found: {args.task}", file=sys.stderr)
            sys.exit(2)

        task = json.loads(args.task.read_text(encoding="utf-8"))

        if args.output_file:
            candidate = args.output_file.read_text(encoding="utf-8")
        elif args.output:
            candidate = args.output
        else:
            candidate = task.get("candidate_output", "")
            if not candidate:
                print("ERROR: provide --output or --output-file", file=sys.stderr)
                sys.exit(2)

        result = score_task(task, candidate)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            status = "PASS" if result.get("pass") else "FAIL"
            print(f"\n{'='*60}")
            print(f"  Task:      {result.get('task_id')}")
            print(f"  Dimension: {result.get('dimension')}")
            print(f"  Score:     {result.get('score')}/{result.get('max_score')} "
                  f"({result.get('normalised_score', 0):.1%})")
            print(f"  Status:    {status}")
            print(f"  Config:    {_DEFAULT_CONFIG.name}")
            print(f"  Notes:")
            for note in result.get("notes", []):
                print(f"    * {note}")
            print(f"{'='*60}\n")

        sys.exit(0 if result.get("pass") else 1)

    else:
        if not args.batch_dir.exists():
            print(f"ERROR: batch dir not found: {args.batch_dir}", file=sys.stderr)
            sys.exit(2)

        summary = batch_score(args.batch_dir)
        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            print(f"\n{'='*60}")
            print(f"  Batch:   {summary['batch_dir']}")
            print(f"  Config:  {_DEFAULT_CONFIG.name}")
            print(f"  Scored:  {summary['scored']}  |  Passed: {summary['passed']}  |  "
                  f"Failed: {summary['failed']}")
            rate = summary.get("pass_at_1")
            print(f"  Pass@1:  {rate:.1%}" if rate is not None else "  Pass@1:  N/A")
            print(f"{'='*60}\n")
            for r in summary["results"]:
                if "error" in r:
                    print(f"  [ERROR] {r['task_id']}: {r['error']}")
                    continue
                status = "+" if r.get("pass") else "-"
                print(f"  [{status}] {r.get('task_id')} ({r.get('dimension')}) "
                      f"{r.get('score')}/{r.get('max_score')}")

        sys.exit(0 if summary.get("pass_at_1", 0) == 1.0 else 1)


if __name__ == "__main__":
    main()
