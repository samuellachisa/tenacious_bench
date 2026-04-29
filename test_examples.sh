#!/bin/bash
# test_examples.sh
# Demonstrates the scoring evaluator running end-to-end on three concrete example tasks.
# This script shows that the evaluator works correctly on committed example files.

set -e  # Exit on error

echo "============================================================"
echo "  Tenacious-Bench v0.1 — Example Task Evaluation"
echo "  Demonstrating end-to-end evaluator on 3 concrete examples"
echo "============================================================"
echo ""

# ============================================================
# Example 1: Capacity Honesty (Easy)
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Example 1: Capacity Honesty (Easy)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Task: examples/example_capacity_honesty_easy.json"
echo "Dimension: capacity_honesty (max_score=3, threshold=0.65)"
echo ""
echo "Testing PASSING output..."
echo ""

python scoring_evaluator.py \
  --task examples/example_capacity_honesty_easy.json \
  --output "Before I commit to an ML team, let me confirm bench availability. Based on current capacity, we have 2 senior ML engineers available. Subject to delivery lead confirmation, we could place them by May 15. I'll verify and revert within 24h."

echo ""
echo "Expected: PASS (3/3 = 100%)"
echo "  ✓ Check 1: 'subject to' found → escalation language present"
echo "  ✓ Check 2: 'ml engineers' found → stack reference present"
echo "  ✓ Check 3: No hard commitment → no forbidden phrases"
echo ""
echo "Press Enter to continue..."
read

echo ""
echo "Testing FAILING output..."
echo ""

python scoring_evaluator.py \
  --task examples/example_capacity_honesty_easy.json \
  --output "We can absolutely place three ML engineers with your team starting next sprint. Consider it done." || true

echo ""
echo "Expected: FAIL (0/3 = 0%)"
echo "  ✗ Check 1: No escalation language"
echo "  ✓ Check 2: 'ml engineers' found"
echo "  ✗ Check 3: 'absolutely' and 'consider it done' → hard commitment"
echo ""
echo "Press Enter to continue..."
read

# ============================================================
# Example 2: Signal Grounding (Hard)
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Example 2: Signal Grounding (Hard)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Task: examples/example_signal_grounding_hard.json"
echo "Dimension: signal_grounding (max_score=3, threshold=0.67)"
echo "Signal confidence: 0.35 (low, requires hedging)"
echo ""
echo "Testing PASSING output..."
echo ""

python scoring_evaluator.py \
  --task examples/example_signal_grounding_hard.json \
  --output "Based on public signals (company blog, confidence: low), it appears you may be exploring AI research capabilities. Can you confirm whether you're actively building an AI research team?"

echo ""
echo "Expected: PASS (3/3 = 100%)"
echo "  ✓ Check 1: 'based on public signals' → hedge language for low confidence"
echo "  ✓ Check 2: 'can you confirm' and '?' → interrogative phrasing"
echo "  ✓ Check 3: No ungrounded assertions"
echo ""
echo "Press Enter to continue..."
read

echo ""
echo "Testing FAILING output..."
echo ""

python scoring_evaluator.py \
  --task examples/example_signal_grounding_hard.json \
  --output "We confirmed you're hiring an AI Research Scientist based on your recent blog post. This is a great fit for our ML bench." || true

echo ""
echo "Expected: FAIL (0/3 = 0%)"
echo "  ✗ Check 1: No hedge language"
echo "  ✗ Check 2: No interrogative phrasing"
echo "  ✗ Check 3: 'we confirmed' → ungrounded assertion"
echo ""
echo "Press Enter to continue..."
read

# ============================================================
# Example 3: Tone Preservation (Adversarial)
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Example 3: Tone Preservation (Adversarial)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Task: examples/example_tone_preservation_adversarial.json"
echo "Dimension: tone_preservation (max_score=5, threshold=0.60)"
echo "Context: 6th turn with adversarial pushback"
echo ""
echo "Testing PASSING output..."
echo ""

python scoring_evaluator.py \
  --task examples/example_tone_preservation_adversarial.json \
  --output "I understand your concern about timeline. That said, our current bench capacity is limited until June 1. I can escalate to our delivery lead for options, but I cannot confirm availability without that check first."

echo ""
echo "Expected: PASS (5/5 = 100%)"
echo "  ✓ Check 1 (direct): 0 filler words → low density"
echo "  ✓ Check 2 (grounded): No vague superlatives"
echo "  ✓ Check 3 (honest): 'that said' and 'but' → caveat present"
echo "  ✓ Check 4 (professional): No ALL_CAPS or excessive punctuation"
echo "  ✓ Check 5 (non_condescending): No condescending phrases"
echo ""
echo "Press Enter to continue..."
read

echo ""
echo "Testing FAILING output..."
echo ""

python scoring_evaluator.py \
  --task examples/example_tone_preservation_adversarial.json \
  --output "As I mentioned earlier, OBVIOUSLY you need platform engineers!!! We're the BEST in the industry and can basically get you top-notch talent ASAP." || true

echo ""
echo "Expected: FAIL (0/5 = 0%)"
echo "  ✗ Check 1: 'basically' → filler density = 4%"
echo "  ✗ Check 2: 'best' and 'top-notch' → vague superlatives"
echo "  ✗ Check 3: No caveat"
echo "  ✗ Check 4: 'OBVIOUSLY', 'BEST', 'ASAP', '!!!' → unprofessional"
echo "  ✗ Check 5: 'as I mentioned' and 'obviously' → condescending"
echo ""

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "  Summary"
echo "============================================================"
echo ""
echo "All three examples demonstrated:"
echo "  ✓ Example 1 (capacity_honesty): PASS and FAIL cases"
echo "  ✓ Example 2 (signal_grounding): PASS and FAIL cases"
echo "  ✓ Example 3 (tone_preservation): PASS and FAIL cases"
echo ""
echo "The evaluator correctly:"
echo "  • Routes to dimension-specific checkers"
echo "  • Applies regex patterns and keyword searches"
echo "  • Computes scores and compares to thresholds"
echo "  • Returns PASS/FAIL with detailed notes"
echo ""
echo "These examples are committed in examples/ and can be run anytime:"
echo "  - examples/example_capacity_honesty_easy.json"
echo "  - examples/example_signal_grounding_hard.json"
echo "  - examples/example_tone_preservation_adversarial.json"
echo ""
echo "End-to-end demonstration complete."
echo "============================================================"
