"""
training/train_simpo_hf.py
Tenacious-Bench v0.1 - SimPO LoRA training via standard HuggingFace PEFT + TRL.
No Unsloth dependency. Works on any Colab T4 environment.

Usage:
    python training/train_simpo_hf.py --pairs training_data/pairs.jsonl --dry-run
    python training/train_simpo_hf.py --pairs training_data/pairs.jsonl --output-dir training/lora_adapter
"""

import argparse
import json
import sys
from pathlib import Path


def dry_run(pairs_path: Path, args) -> None:
    pairs = []
    with open(pairs_path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                pairs.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"ERROR line {i}: {e}", file=sys.stderr)
                sys.exit(1)

    print(f"\n[DRY RUN] SimPO Training Plan (HuggingFace PEFT, no Unsloth)")
    print("=" * 60)
    print(f"  Pairs file : {pairs_path}  ({len(pairs)} pairs)")
    print(f"  Base model : {args.model}")
    print(f"  Output dir : {args.output_dir}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  LR         : {args.lr}")
    print(f"  Batch size : {args.batch_size} (grad accum {args.grad_accum})")
    print(f"  LoRA rank  : {args.lora_rank}  (alpha={args.lora_rank * 2})")
    print(f"  SimPO gamma: {args.simpo_gamma}")
    print(f"  SimPO beta : {args.simpo_beta}")

    required = {"pair_id", "dimension", "chosen", "rejected"}
    bad = 0
    for p in pairs:
        missing = required - set(p.keys())
        if missing:
            print(f"  WARNING pair {p.get('pair_id','?')} missing: {missing}")
            bad += 1
        if "output" not in p.get("chosen", {}):
            print(f"  WARNING pair {p.get('pair_id','?')} chosen.output missing")
            bad += 1
        if "output" not in p.get("rejected", {}):
            print(f"  WARNING pair {p.get('pair_id','?')} rejected.output missing")
            bad += 1

    dim_counts: dict = {}
    for p in pairs:
        d = p.get("dimension", "unknown")
        dim_counts[d] = dim_counts.get(d, 0) + 1

    print(f"\n  Dimension distribution:")
    for d, c in sorted(dim_counts.items()):
        print(f"    {d}: {c} pairs ({c/len(pairs):.0%})")

    if bad == 0:
        print(f"\n[DRY RUN] Structure check passed. Run without --dry-run to train.")
    else:
        print(f"\n[DRY RUN] {bad} warnings found. Fix before training.")
    print("=" * 60)
    sys.exit(0)


def run_training(args) -> None:
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, TaskType
        from trl import CPOConfig, CPOTrainer
        from datasets import Dataset
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}", file=sys.stderr)
        print("Install with: pip install trl transformers datasets peft bitsandbytes accelerate", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Loading model: {args.model} (4-bit quantized)")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    print(f"Applying LoRA (rank={args.lora_rank})")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"Loading pairs from {args.pairs}")
    records = []
    with open(args.pairs, encoding="utf-8") as f:
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
            })

    dataset = Dataset.from_list(records)
    split = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Train: {len(train_dataset)} pairs  |  Eval: {len(eval_dataset)} pairs")

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
        eval_strategy="steps",
        loss_type="simpo",
        beta=args.simpo_beta,
        cpo_alpha=args.simpo_gamma,
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = CPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    print("Starting SimPO training...")
    trainer.train()

    print(f"Saving LoRA adapter to {output_dir}")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

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
    print(f"Training complete. Summary saved to {output_dir}/training_summary.json")


def main():
    parser = argparse.ArgumentParser(
        description="Tenacious-Bench SimPO LoRA training (HuggingFace PEFT, no Unsloth)"
    )
    parser.add_argument("--pairs", default="training_data/pairs.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model ID")
    parser.add_argument("--output-dir", default="training/lora_adapter")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--simpo-gamma", type=float, default=0.5)
    parser.add_argument("--simpo-beta", type=float, default=2.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    pairs_path = Path(args.pairs)
    if not pairs_path.exists():
        print(f"ERROR: pairs file not found: {pairs_path}", file=sys.stderr)
        sys.exit(2)

    if args.dry_run:
        dry_run(pairs_path, args)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
