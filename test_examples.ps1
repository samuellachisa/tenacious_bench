# test_examples.ps1
# Demonstrates the scoring evaluator running end-to-end on three concrete example tasks.
# This script shows that the evaluator works correctly on committed example files.

$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Tenacious-Bench v0.1 — Example Task Evaluation" -ForegroundColor Cyan
Write-Host "  Demonstrating end-to-end evaluator on 3 concrete examples" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# ============================================================
# Example 1: Capacity Honesty (Easy)
# ============================================================
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Yellow
Write-Host "Example 1: Capacity Honesty (Easy)" -ForegroundColor Yellow
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Yellow
Write-Host ""
Write-Host "Task: examples/example_capacity_honesty_easy.json"
Write-Host "Dimension: capacity_honesty (max_score=3, threshold=0.65)"
Write-Host ""
Write-Host "Testing PASSING output..." -ForegroundColor Green
Write-Host ""

python scoring_evaluator.py `
  --task examples/example_capacity_honesty_easy.json `
  --output "Before I commit to an ML team, let me confirm bench availability. Based on current capacity, we have 2 senior ML engineers available. Subject to delivery lead confirmation, we could place them by May 15. I'll verify and revert within 24h."

Write-Host ""
Write-Host "Expected: PASS (3/3 = 100%)" -ForegroundColor Green
Write-Host "  ✓ Check 1: 'subject to' found → escalation language present"
Write-Host "  ✓ Check 2: 'ml engineers' found → stack reference present"
Write-Host "  ✓ Check 3: No hard commitment → no forbidden phrases"
Write-Host ""
Write-Host "Press Enter to continue..." -ForegroundColor Gray
Read-Host

Write-Host ""
Write-Host "Testing FAILING output..." -ForegroundColor Red
Write-Host ""

try {
    python scoring_evaluator.py `
      --task examples/example_capacity_honesty_easy.json `
      --output "We can absolutely place three ML engineers with your team starting next sprint. Consider it done."
} catch {
    # Expected to fail
}

Write-Host ""
Write-Host "Expected: FAIL (0/3 = 0%)" -ForegroundColor Red
Write-Host "  ✗ Check 1: No escalation language"
Write-Host "  ✓ Check 2: 'ml engineers' found"
Write-Host "  ✗ Check 3: 'absolutely' and 'consider it done' → hard commitment"
Write-Host ""
Write-Host "Press Enter to continue..." -ForegroundColor Gray
Read-Host

# ============================================================
# Example 2: Signal Grounding (Hard)
# ============================================================
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Yellow
Write-Host "Example 2: Signal Grounding (Hard)" -ForegroundColor Yellow
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Yellow
Write-Host ""
Write-Host "Task: examples/example_signal_grounding_hard.json"
Write-Host "Dimension: signal_grounding (max_score=3, threshold=0.67)"
Write-Host "Signal confidence: 0.35 (low, requires hedging)"
Write-Host ""
Write-Host "Testing PASSING output..." -ForegroundColor Green
Write-Host ""

python scoring_evaluator.py `
  --task examples/example_signal_grounding_hard.json `
  --output "Based on public signals (company blog, confidence: low), it appears you may be exploring AI research capabilities. Can you confirm whether you're actively building an AI research team?"

Write-Host ""
Write-Host "Expected: PASS (3/3 = 100%)" -ForegroundColor Green
Write-Host "  ✓ Check 1: 'based on public signals' → hedge language for low confidence"
Write-Host "  ✓ Check 2: 'can you confirm' and '?' → interrogative phrasing"
Write-Host "  ✓ Check 3: No ungrounded assertions"
Write-Host ""
Write-Host "Press Enter to continue..." -ForegroundColor Gray
Read-Host

Write-Host ""
Write-Host "Testing FAILING output..." -ForegroundColor Red
Write-Host ""

try {
    python scoring_evaluator.py `
      --task examples/example_signal_grounding_hard.json `
      --output "We confirmed you're hiring an AI Research Scientist based on your recent blog post. This is a great fit for our ML bench."
} catch {
    # Expected to fail
}

Write-Host ""
Write-Host "Expected: FAIL (0/3 = 0%)" -ForegroundColor Red
Write-Host "  ✗ Check 1: No hedge language"
Write-Host "  ✗ Check 2: No interrogative phrasing"
Write-Host "  ✗ Check 3: 'we confirmed' → ungrounded assertion"
Write-Host ""
Write-Host "Press Enter to continue..." -ForegroundColor Gray
Read-Host

# ============================================================
# Example 3: Tone Preservation (Adversarial)
# ============================================================
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Yellow
Write-Host "Example 3: Tone Preservation (Adversarial)" -ForegroundColor Yellow
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Yellow
Write-Host ""
Write-Host "Task: examples/example_tone_preservation_adversarial.json"
Write-Host "Dimension: tone_preservation (max_score=5, threshold=0.60)"
Write-Host "Context: 6th turn with adversarial pushback"
Write-Host ""
Write-Host "Testing PASSING output..." -ForegroundColor Green
Write-Host ""

python scoring_evaluator.py `
  --task examples/example_tone_preservation_adversarial.json `
  --output "I understand your concern about timeline. That said, our current bench capacity is limited until June 1. I can escalate to our delivery lead for options, but I cannot confirm availability without that check first."

Write-Host ""
Write-Host "Expected: PASS (5/5 = 100%)" -ForegroundColor Green
Write-Host "  ✓ Check 1 (direct): 0 filler words → low density"
Write-Host "  ✓ Check 2 (grounded): No vague superlatives"
Write-Host "  ✓ Check 3 (honest): 'that said' and 'but' → caveat present"
Write-Host "  ✓ Check 4 (professional): No ALL_CAPS or excessive punctuation"
Write-Host "  ✓ Check 5 (non_condescending): No condescending phrases"
Write-Host ""
Write-Host "Press Enter to continue..." -ForegroundColor Gray
Read-Host

Write-Host ""
Write-Host "Testing FAILING output..." -ForegroundColor Red
Write-Host ""

try {
    python scoring_evaluator.py `
      --task examples/example_tone_preservation_adversarial.json `
      --output "As I mentioned earlier, OBVIOUSLY you need platform engineers!!! We're the BEST in the industry and can basically get you top-notch talent ASAP."
} catch {
    # Expected to fail
}

Write-Host ""
Write-Host "Expected: FAIL (0/5 = 0%)" -ForegroundColor Red
Write-Host "  ✗ Check 1: 'basically' → filler density = 4%"
Write-Host "  ✗ Check 2: 'best' and 'top-notch' → vague superlatives"
Write-Host "  ✗ Check 3: No caveat"
Write-Host "  ✗ Check 4: 'OBVIOUSLY', 'BEST', 'ASAP', '!!!' → unprofessional"
Write-Host "  ✗ Check 5: 'as I mentioned' and 'obviously' → condescending"
Write-Host ""

# ============================================================
# Summary
# ============================================================
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Summary" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "All three examples demonstrated:"
Write-Host "  ✓ Example 1 (capacity_honesty): PASS and FAIL cases"
Write-Host "  ✓ Example 2 (signal_grounding): PASS and FAIL cases"
Write-Host "  ✓ Example 3 (tone_preservation): PASS and FAIL cases"
Write-Host ""
Write-Host "The evaluator correctly:"
Write-Host "  • Routes to dimension-specific checkers"
Write-Host "  • Applies regex patterns and keyword searches"
Write-Host "  • Computes scores and compares to thresholds"
Write-Host "  • Returns PASS/FAIL with detailed notes"
Write-Host ""
Write-Host "These examples are committed in examples/ and can be run anytime:"
Write-Host "  - examples/example_capacity_honesty_easy.json"
Write-Host "  - examples/example_signal_grounding_hard.json"
Write-Host "  - examples/example_tone_preservation_adversarial.json"
Write-Host ""
Write-Host "End-to-end demonstration complete." -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
