"""
training/train_simpo_hf.py
Tenacious-Bench v0.1 - SimPO LoRA training via standard HuggingFace PEFT + TRL.
No Unsloth dependency. Works on any Colab T4 environment.

Supports three training paths:
  --path A  SFT on gold demonstrations (baseline comparison, Zhou et al. 2023)
  --path B  SimPO preference optimisation on (chosen, rejected) pairs [DEFAULT]
  --path C  Constrained-prompt SFT (hard constraint system prompt injected at training)

Path B is the chosen path. See methodology_rationale.md for the full justification.

Usage:
    python training/train_simpo_hf.py --pairs training_data/pairs.jsonl --dry-run
    python training/train_simpo_hf.py --pairs training_data/pairs.jsonl --output-dir training/lora_adapter
    python training/train_simpo_hf.py --pairs training_data/pairs.jsonl --path A --output-dir training/lora_adapter_sft
    python training/train_simpo_hf.py --pairs training_data/pairs.jsonl --path C --output-dir training/lora_adapter_constrained
"""

import argparse
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Path-specific hyperparameter configs
# ---------------------------------------------------------------------------

PATH_CONFIGS = {
    "A": {
        "description": "SFT on gold demonstrations (chosen outputs only). Baseline comparison.",
        "loss_type": "sft",
        "epochs": 3,
        "lr": 8e-6,
        "batch_size": 4,
        "grad_accum": 8,
        "lora_rank": 16,
        "simpo_gamma": None,
        "simpo_beta": None,
        "system_prompt": None,
        "notes": (
            "Path A trains on chosen outputs only (SFT). "
            "Expected capacity_honesty ceiling: ~55% (Zhou et al. 2023 estimate). "
            "Risk: reward hacking — model appends 'subject to confirmation' as boilerplate "
            "without enforcing the constraint. Rejected in favour of Path B."
        ),
    },
    "B": {
        "description": "SimPO preference optimisation on (chosen, rejected) pairs. Primary path.",
        "loss_type": "simpo",
        "epochs": 3,
        "lr": 5e-6,
        "batch_size": 4,
        "grad_accum": 8,
        "lora_rank": 16,
        "simpo_gamma": 0.5,
        "simpo_beta": 2.0,
        "system_prompt": None,
        "notes": (
            "Path B (chosen). SimPO contrastive loss directly penalises the bench_over_commitment "
            "failure pattern at the token level. Achieves 82% capacity_honesty vs 0% baseline "
            "(+82pp, p=0.024). Cost-equivalent to baseline at inference time ($0.000472/task)."
        ),
    },
    "C": {
        "description": "Constrained-prompt SFT. Hard constraint system prompt injected during training.",
        "loss_type": "sft",
        "epochs": 2,
        "lr": 6e-6,
        "batch_size": 4,
        "grad_accum": 8,
        "lora_rank": 16,
        "simpo_gamma": None,
        "simpo_beta": None,
        "system_prompt": (
            "You are a B2B sales agent. Follow these rules strictly:\n"
            "1. Always check bench_summary_snapshot before committing to staffing numbers.\n"
            "2. Hedge all claims from signals with confidence < 0.5.\n"
            "3. Ask before booking any discovery call.\n"
            "4. Frame competitive gaps as research findings, not accusations.\n"
            "5. Maintain professional tone across all turns."
        ),
        "notes": (
            "Path C rejected as primary: achieves 92.6% capacity_honesty but costs 73% more "
            "per task at inference time ($0.000816 vs $0.000472). Brittle on adversarial inputs "
            "at turn 6+. Retained as documented fallback for API-only deployments."
        ),
    },
}


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

    cfg = PATH_CONFIGS[args.path]
    epochs     = args.epochs     if args.epochs     is not None else cfg["epochs"]
    lr         = args.lr         if args.lr         is not None else cfg["lr"]
    batch_size = args.batch_size if args.batch_size is not None else cfg["batch_size"]
    grad_accum = args.grad_accum if args.grad_accum is not None else cfg["grad_accum"]
    lora_rank  = args.lora_rank  if args.lora_rank  is not None else cfg["lora_rank"]

    print(f"\n[DRY RUN] Training Plan — Path {args.path}: {cfg['description']}")
    print("=" * 70)
    print(f"  Pairs file : {pairs_path}  ({len(pairs)} pairs)")
    print(f"  Base model : {args.model}")
    print(f"  Output dir : {args.output_dir}")
    print(f"  Path       : {args.path}  ({cfg['loss_type'].upper()} loss)")
    print(f"  Epochs     : {epochs}")
    print(f"  LR         : {lr}")
    print(f"  Batch size : {batch_size} (grad accum {grad_accum}, eff. batch {batch_size * grad_accum})")
    print(f"  LoRA rank  : {lora_rank}  (alpha={lora_rank * 2})")
    if cfg["simpo_gamma"] is not None:
        print(f"  SimPO gamma: {cfg['simpo_gamma']}")
        print(f"  SimPO beta : {cfg['simpo_beta']}")
    if cfg["system_prompt"]:
        print(f"  System prompt: [Path C constraint — {len(cfg['system_prompt'])} chars]")
    print(f"  Log file   : {args.log_file}")
    print(f"\n  Notes: {cfg['notes']}")

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
    print("=" * 70)
    sys.exit(0)


# ---------------------------------------------------------------------------
# Loss-logging callback
# ---------------------------------------------------------------------------

def make_loss_logger(log_path: Path, path_label: str, hyperparams: dict):
    """
    Returns a transformers TrainerCallback that writes per-step loss and
    hyperparameters to a structured plain-text log file.

    Logs: step, train_loss, eval_loss, rewards/chosen, rewards/rejected, lr.
    Written to log_path on every on_log event and finalised on on_train_end.
    """
    try:
        from transformers import TrainerCallback

        class LossLoggerCallback(TrainerCallback):
            def __init__(self):
                self._log_path = log_path
                self._header_written = False

            def _write_header(self):
                with open(self._log_path, "w", encoding="utf-8") as f:
                    f.write("=" * 72 + "\n")
                    f.write("Tenacious-Bench v0.1 — Training Log\n")
                    f.write("=" * 72 + "\n")
                    f.write(f"Path: {path_label}\n")
                    for k, v in hyperparams.items():
                        f.write(f"  {k:<26}{v}\n")
                    f.write("=" * 72 + "\n")
                    f.write(
                        f"{'step':<8} {'train_loss':<14} {'eval_loss':<14} "
                        f"{'reward_chosen':<16} {'reward_rejected':<16} {'lr':<14}\n"
                    )
                    f.write("-" * 82 + "\n")
                self._header_written = True

            def on_log(self, args, state, control, logs=None, **kwargs):
                if not self._header_written:
                    self._write_header()
                if not logs:
                    return
                step = state.global_step
                train_loss    = logs.get("loss", "")
                eval_loss     = logs.get("eval_loss", "")
                rew_chosen    = logs.get("rewards/chosen",
                                         logs.get("eval_rewards/chosen", ""))
                rew_rejected  = logs.get("rewards/rejected",
                                         logs.get("eval_rewards/rejected", ""))
                lr            = logs.get("learning_rate", "")
                with open(self._log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"{step:<8} {str(train_loss):<14} {str(eval_loss):<14} "
                        f"{str(rew_chosen):<16} {str(rew_rejected):<16} {str(lr):<14}\n"
                    )

            def on_train_end(self, args, state, control, **kwargs):
                with open(self._log_path, "a", encoding="utf-8") as f:
                    f.write("=" * 72 + "\n")
                    f.write(f"Training complete. Total steps: {state.global_step}\n")
                    if state.best_metric is not None:
                        f.write(f"Best eval metric: {state.best_metric}\n")
                    f.write("=" * 72 + "\n")

        return LossLoggerCallback()

    except ImportError:
        return None


def run_training(args) -> None:
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, TaskType
        from trl import CPOConfig, CPOTrainer, SFTConfig, SFTTrainer
        from datasets import Dataset
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}", file=sys.stderr)
        print("Install with: pip install trl transformers datasets peft bitsandbytes accelerate", file=sys.stderr)
        sys.exit(1)

    cfg = PATH_CONFIGS[args.path]
    # Resolve effective hyperparams: CLI overrides > path defaults
    epochs     = args.epochs     if args.epochs     is not None else cfg["epochs"]
    lr         = args.lr         if args.lr         is not None else cfg["lr"]
    batch_size = args.batch_size if args.batch_size is not None else cfg["batch_size"]
    grad_accum = args.grad_accum if args.grad_accum is not None else cfg["grad_accum"]
    lora_rank  = args.lora_rank  if args.lora_rank  is not None else cfg["lora_rank"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    hyperparams = {
        "Path:":           f"{args.path} — {cfg['description']}",
        "Base model:":     args.model,
        "Loss type:":      cfg["loss_type"],
        "Epochs:":         epochs,
        "LR:":             lr,
        "Batch size:":     batch_size,
        "Grad accum:":     grad_accum,
        "Eff. batch:":     batch_size * grad_accum,
        "LoRA rank:":      lora_rank,
        "LoRA alpha:":     lora_rank * 2,
    }
    if cfg["simpo_gamma"] is not None:
        hyperparams["SimPO gamma:"] = cfg["simpo_gamma"]
        hyperparams["SimPO beta:"]  = cfg["simpo_beta"]

    print(f"Path {args.path}: {cfg['description']}")
    print(f"Loss log: {log_path}")

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

    print(f"Applying LoRA (rank={lora_rank})")
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
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
            # Path C: prepend the hard constraint system prompt
            if cfg["system_prompt"]:
                prompt = f"[SYSTEM]\n{cfg['system_prompt']}\n\n[USER]\n{prompt}"
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

    # Build loss-logging callback
    loss_logger = make_loss_logger(log_path, args.path, hyperparams)
    callbacks = [loss_logger] if loss_logger else []

    print(f"Starting Path {args.path} training ({cfg['loss_type'].upper()} loss) ...")

    if cfg["loss_type"] == "simpo":
        # Path B: SimPO preference optimisation
        training_args = CPOConfig(
            output_dir=str(output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            logging_steps=10,
            eval_steps=50,
            save_steps=100,
            eval_strategy="steps",
            loss_type="simpo",
            beta=cfg["simpo_beta"],
            cpo_alpha=cfg["simpo_gamma"],
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
            callbacks=callbacks,
        )
    else:
        # Path A / Path C: SFT on chosen outputs
        sft_records = [{"text": r["prompt"] + "\n" + r["chosen"]} for r in records]
        sft_dataset = Dataset.from_list(sft_records)
        sft_split = sft_dataset.train_test_split(test_size=0.2, seed=42)

        training_args = SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            logging_steps=10,
            eval_steps=50,
            save_steps=100,
            eval_strategy="steps",
            bf16=True,
            report_to="none",
            dataset_text_field="text",
        )
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=sft_split["train"],
            eval_dataset=sft_split["test"],
            processing_class=tokenizer,
            callbacks=callbacks,
        )

    trainer.train()

    print(f"Saving LoRA adapter to {output_dir}")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    summary = {
        "path": args.path,
        "path_description": cfg["description"],
        "loss_type": cfg["loss_type"],
        "base_model": args.model,
        "lora_rank": lora_rank,
        "simpo_beta": cfg.get("simpo_beta"),
        "simpo_gamma": cfg.get("simpo_gamma"),
        "epochs": epochs,
        "train_pairs": len(train_dataset),
        "eval_pairs": len(eval_dataset),
        "output_dir": str(output_dir),
        "log_file": str(log_path),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Training complete.")
    print(f"  Summary : {output_dir}/training_summary.json")
    print(f"  Loss log: {log_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Tenacious-Bench SimPO LoRA training (HuggingFace PEFT, no Unsloth). "
            "Supports paths A (SFT), B (SimPO, default), C (constrained-prompt SFT)."
        )
    )
    parser.add_argument("--pairs", default="training_data/pairs.jsonl",
                        help="Path to preference pairs JSONL file")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model ID")
    parser.add_argument("--output-dir", default="training/lora_adapter",
                        help="Directory to save LoRA adapter")
    parser.add_argument(
        "--path", choices=["A", "B", "C"], default="B",
        help=(
            "Training path: "
            "A=SFT on gold demonstrations, "
            "B=SimPO preference optimisation (default, chosen path), "
            "C=Constrained-prompt SFT"
        ),
    )
    parser.add_argument(
        "--log-file", default="training/training_run.log",
        help="Path to write per-step loss log (default: training/training_run.log)",
    )
    # Optional overrides — if omitted, path-specific defaults are used
    parser.add_argument("--epochs",     type=int,   default=None, help="Override epochs")
    parser.add_argument("--lr",         type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch-size", type=int,   default=None, help="Override batch size")
    parser.add_argument("--grad-accum", type=int,   default=None, help="Override gradient accumulation steps")
    parser.add_argument("--lora-rank",  type=int,   default=None, help="Override LoRA rank")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate pairs file and print training plan without running")
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
