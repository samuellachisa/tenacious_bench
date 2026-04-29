"""
training/train_simpo.py
Tenacious-Bench v0.1 — SimPO LoRA training script via Unsloth.

Trains a preference-optimised LoRA adapter on the Tenacious-Bench preference
pairs (training_data/pairs.jsonl) using the SimPO objective.

Prerequisites:
    pip install unsloth trl transformers datasets torch accelerate peft

Usage:
    python training/train_simpo.py \
        --pairs training_data/pairs.jsonl \
        --model unsloth/Qwen3-8B-bnb-4bit \
        --output-dir training/lora_adapter \
        --epochs 3 \
        --lr 5e-6 \
        --batch-size 4 \
        --grad-accum 8 \
        --lora-rank 16 \
        --simpo-gamma 0.5 \
        --simpo-beta 2.0 \
        --eval-dir tenacious_bench_v0.1/dev \
        --budget-usd 10.0

Dry-run (structure check, no GPU required):
    python training/train_simpo.py --dry-run
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Dry-run mode (no GPU, validates structure only)
# ---------------------------------------------------------------------------

def _dry_run(args: argparse.Namespace) -> None:
    """Validate pair file structure and print training plan."""
    pairs_path = Path(args.pairs)
    if not pairs_path.exists():
        print(f"ERROR: pairs file not found: {pairs_path}", file=sys.stderr)
        sys.exit(2)

    pairs = []
    with open(pairs_path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                pair = json.loads(line)
                pairs.append(pair)
            except json.JSONDecodeError as e:
                print(f"ERROR: malformed JSON at line {i}: {e}", file=sys.stderr)
                sys.exit(1)

    print(f"\n[DRY RUN] SimPO Training Plan")
    print(f"{'='*60}")
    print(f"  Pairs file:     {pairs_path}  ({len(pairs)} pairs)")
    print(f"  Base model:     {args.model}")
    print(f"  Output dir:     {args.output_dir}")
    print(f"  Epochs:         {args.epochs}")
    print(f"  LR:             {args.lr}")
    print(f"  Batch size:     {args.batch_size} (grad accum {args.grad_accum})")
    print(f"  Effective batch:{args.batch_size * args.grad_accum}")
    print(f"  LoRA rank:      {args.lora_rank}  (alpha={args.lora_rank * 2})")
    print(f"  SimPO gamma:    {args.simpo_gamma}")
    print(f"  SimPO beta:     {args.simpo_beta}")
    print(f"  Budget cap:     ${args.budget_usd}")

    # Validate pair schema
    required_keys = {"pair_id", "dimension", "chosen", "rejected"}
    for p in pairs[:5]:
        missing = required_keys - set(p.keys())
        if missing:
            print(f"WARNING: pair {p.get('pair_id', '?')} missing keys: {missing}")

    dim_counts: dict[str, int] = {}
    for p in pairs:
        d = p.get("dimension", "unknown")
        dim_counts[d] = dim_counts.get(d, 0) + 1

    print(f"\n  Dimension distribution:")
    for d, c in sorted(dim_counts.items()):
        print(f"    {d}: {c} pairs ({c/len(pairs):.0%})")

    # Estimate training cost (rough)
    tokens_per_pair = 400  # avg chosen+rejected token estimate
    total_tokens = len(pairs) * args.epochs * tokens_per_pair
    print(f"\n  Estimated training tokens: {total_tokens:,}")
    print(f"  Estimated GPU-hours (A100): ~{total_tokens / 10_000_000:.1f}h")
    print(f"\n[DRY RUN] Structure check passed. Run without --dry-run to train.")
    print(f"{'='*60}\n")
    sys.exit(0)


# ---------------------------------------------------------------------------
# Training (requires Unsloth + GPU)
# ---------------------------------------------------------------------------

def _build_simpo_loss_fn(beta: float, gamma: float):
    """
    SimPO loss: L = -log σ(β * (r_chosen - r_rejected) - γ)
    where r = (1/|y|) * sum log p(y_i | x, y_{<i})
    (length-normalised log-probability)

    This is a simplified reference implementation. The real training loop
    uses TRL's CPOTrainer with SimPO config or a custom HuggingFace Trainer.
    """
    try:
        import torch
        import torch.nn.functional as F

        def simpo_loss(chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
            # Length-normalise: logps are already per-token averages from TRL
            delta = chosen_logps - rejected_logps
            loss = -F.logsigmoid(beta * delta - gamma).mean()
            return loss

        return simpo_loss
    except ImportError:
        raise RuntimeError("PyTorch is required for training. Install with: pip install torch")


def _load_pairs_as_dataset(pairs_path: Path, tokenizer: Any) -> Any:
    """
    Load pairs.jsonl and convert to HuggingFace Dataset format expected by TRL.
    TRL CPOTrainer expects: {'prompt': str, 'chosen': str, 'rejected': str}
    """
    try:
        from datasets import Dataset
    except ImportError:
        raise RuntimeError("datasets library required: pip install datasets")

    records = []
    with open(pairs_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            prompt = json.dumps(p.get("input", {}), ensure_ascii=False)
            records.append({
                "prompt": prompt,
                "chosen": p["chosen"]["output"],
                "rejected": p["rejected"]["output"],
                "dimension": p.get("dimension", ""),
            })

    return Dataset.from_list(records)


def _run_training(args: argparse.Namespace) -> None:
    """Full training loop using Unsloth + TRL CPOTrainer (SimPO config)."""
    try:
        from unsloth import FastLanguageModel
        from trl import CPOConfig, CPOTrainer
    except ImportError:
        print(
            "ERROR: Unsloth and TRL are required for training.\n"
            "Install with: pip install unsloth trl",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=2048,
        dtype=None,  # auto-detect
        load_in_4bit=True,
    )

    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    print(f"Loading pairs from {args.pairs}")
    dataset = _load_pairs_as_dataset(Path(args.pairs), tokenizer)
    # 80/20 train/eval split from pairs file
    split = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    print(f"Training on {len(train_dataset)} pairs, eval on {len(eval_dataset)}")

    # CPOTrainer in SimPO mode (loss_type="simpo")
    training_args = CPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        eval_steps=50,
        save_steps=100,
        evaluation_strategy="steps",
        loss_type="simpo",
        beta=args.simpo_beta,
        cpo_alpha=args.simpo_gamma,  # gamma margin in SimPO via CPO alpha
        bf16=True,
        report_to="none",
    )

    trainer = CPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    print("Starting SimPO training...")
    trainer.train()

    print(f"Saving LoRA adapter to {output_dir}")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save training summary
    summary = {
        "base_model": args.model,
        "lora_rank": args.lora_rank,
        "simpo_beta": args.simpo_beta,
        "simpo_gamma": args.simpo_gamma,
        "epochs": args.epochs,
        "train_pairs": len(train_dataset),
        "eval_pairs": len(eval_dataset),
        "output_dir": str(output_dir),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Training complete. Summary: {output_dir / 'training_summary.json'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Tenacious-Bench v0.1 — SimPO LoRA training via Unsloth"
    )
    parser.add_argument("--pairs", default="training_data/pairs.jsonl")
    parser.add_argument("--model", default="unsloth/Qwen3-8B-bnb-4bit",
                        help="HuggingFace model ID (Unsloth 4-bit quantized recommended)")
    parser.add_argument("--output-dir", default="training/lora_adapter")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--simpo-gamma", type=float, default=0.5)
    parser.add_argument("--simpo-beta", type=float, default=2.0)
    parser.add_argument("--eval-dir", default="tenacious_bench_v0.1/dev")
    parser.add_argument("--budget-usd", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate structure only — no GPU required")
    args = parser.parse_args()

    if args.dry_run:
        _dry_run(args)
    else:
        _run_training(args)


if __name__ == "__main__":
    main()
