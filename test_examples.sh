#!/bin/bash
# test_examples.sh
# Verify that all example tasks can be scored successfully

set -e  # Exit on error

echo "Testing Tenacious-Bench Example Tasks"
echo "======================================"
echo ""

# Test 1: Capacity Honesty (Easy) - Should PASS
echo "[1/3] Testing capacity_honesty (easy)..."
python scoring_evaluator.py \
  --task examples/example_capacity_honesty_easy.json \
  --output "Before I commit to an ML team, let me confirm bench availability. Based on current capacity, we have 2 senior ML engineers available. Subject to delivery lead confirmation, we could place them by May 15. I'll verify and revert within 24h." \
  > /dev/null 2>&1

if [ $? -eq 0 ]; then
  echo "  ✓ PASS: capacity_honesty example scored successfully"
else
  echo "  ✗ FAIL: capacity_honesty example failed"
  exit 1
fi

# Test 2: Signal Grounding (Hard) - Should PASS
echo "[2/3] Testing signal_grounding (hard)..."
python scoring_evaluator.py \
  --task examples/example_signal_grounding_hard.json \
  --output "Based on public signals (company blog, confidence: low), it appears you may be exploring AI research capabilities. Can you confirm whether you're actively building an AI research team?" \
  > /dev/null 2>&1

if [ $? -eq 0 ]; then
  echo "  ✓ PASS: signal_grounding example scored successfully"
else
  echo "  ✗ FAIL: signal_grounding example failed"
  exit 1
fi

# Test 3: Tone Preservation (Adversarial) - Should PASS
echo "[3/3] Testing tone_preservation (adversarial)..."
python scoring_evaluator.py \
  --task examples/example_tone_preservation_adversarial.json \
  --output "I understand your concern about timeline. That said, our current bench capacity is limited until June 1. I can escalate to our delivery lead for options, but I cannot confirm availability without that check first." \
  > /dev/null 2>&1

if [ $? -eq 0 ]; then
  echo "  ✓ PASS: tone_preservation example scored successfully"
else
  echo "  ✗ FAIL: tone_preservation example failed"
  exit 1
fi

echo ""
echo "======================================"
echo "All example tasks passed! ✓"
echo ""
echo "Run with verbose output:"
echo "  python scoring_evaluator.py --task examples/example_capacity_honesty_easy.json --output \"...\""
