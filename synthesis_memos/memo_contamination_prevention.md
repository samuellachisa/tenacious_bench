# Synthesis Memo — Recent Advances in LLM Benchmarks Against Data Contamination
**Common Reading 3 | Chen et al., EMNLP 2025**

---

## Paper Summary

Chen et al. survey contamination-prevention techniques across static and dynamic evaluation paradigms. The core finding: static benchmarks (fixed task sets) are contaminated within 6–18 months of public release because LLM pretraining corpora are updated continuously. The paper proposes a taxonomy of contamination types — **direct contamination** (exact task text in pretraining), **indirect contamination** (paraphrased or semantically similar text), and **temporal contamination** (tasks reference events that postdate the model's knowledge cutoff but are in the training data via data leakage).

The paper's recommended defenses: n-gram overlap checks (direct), embedding similarity checks (indirect), time-shift verification (temporal), and dynamic task generation (structural defense).

---

## Application to Tenacious-Bench

### Where I agree

**N-gram + cosine is the right two-layer check.** The `contamination_check.py` implements exactly the paper's recommended defense: 8-gram overlap for direct contamination, TF-IDF cosine similarity (threshold 0.85) for indirect contamination. The paper's empirical finding that 8-gram overlap is the right granularity for task-level contamination (not 4-gram, which produces too many false positives) informed the threshold choice.

**Dynamic generation is the structural defense.** The programmatic sweep in `generation_scripts/generate_dataset.py` (seed=42, parameter combinations) produces tasks that are structurally novel even if individual fixture values appear in pretraining data. The adversarial tier (hand-authored) provides the strongest contamination resistance because it was written specifically to defeat the Week 10 agent, not to match any public benchmark pattern.

**Temporal contamination is real for Tenacious-Bench.** The bench_summary.json snapshot is dated 2026-04-21. Any task that references specific capacity numbers from this snapshot is temporally grounded — a model trained on data after this date could have seen the snapshot. The contamination check's time-shift verification flag in the task metadata (`metadata.created_at`) documents this.

### Where I disagree

**The paper recommends embedding-based contamination checks using large embedding models (e.g., text-embedding-3-large).** I used TF-IDF cosine similarity instead. The paper's recommendation is correct for general-purpose benchmarks where semantic paraphrase is the primary contamination vector. For Tenacious-Bench, the contamination risk is structural (same bench_summary.json values, same probe patterns) rather than semantic. TF-IDF captures structural overlap more reliably than dense embeddings for this domain, and it requires no API calls or external dependencies.

**Evidence:** The contamination check ran in under 2 seconds on 250 tasks with TF-IDF. A dense embedding check would require ~250 API calls at ~$0.002 each = $0.50, and would flag false positives on tasks that share domain vocabulary (e.g., "bench", "capacity", "ML engineer") without being contaminated. The TF-IDF approach is cheaper, faster, and more appropriate for this domain.

**The paper also recommends dynamic task generation as the primary defense.** I agree in principle but note that dynamic generation introduces its own quality risk: tasks generated on-the-fly at eval time may be inconsistent or easier than the static held-out set. For a benchmark that needs to be reproducible (a stranger can clone and reproduce the headline number in under an hour), static generation with a fixed seed is the right tradeoff.

---

## Key Design Decisions Informed by This Paper

1. `contamination_check.py` implements 8-gram overlap + TF-IDF cosine (threshold 0.85).
2. Held-out partition is sealed in a separate directory and gitignored from training scripts.
3. All tasks include `metadata.created_at` for temporal contamination documentation.
4. Adversarial tier (hand-authored) provides structural contamination resistance.

---

## How I Operationalized My Alternative Design

My alternative to the paper's dense-embedding recommendation: TF-IDF cosine similarity as the indirect contamination check, with time-shift verification as the temporal layer.

- [x] **TF-IDF cosine implemented with no external dependencies.** `contamination_check.py` `run_check()` builds IDF from the full task corpus + reference texts using only Python stdlib (`math`, `collections.Counter`). No API calls, no sklearn, runs in <2 seconds on 250 tasks.
- [x] **8-gram threshold calibrated to domain.** `ngram_overlap()` uses n=8 (not 4 or 6) following Chen et al.'s empirical finding that 8-gram is the right granularity for task-level contamination. The threshold is exposed as `--ngram` CLI arg for reproducibility.
- [x] **Cosine threshold 0.85 with documented false-positive resolution.** `methodology.md` §Contamination Check Results documents 12 initial flags: 7 false positives (shared domain vocabulary), 3 intentional trace-derived tasks (retained), 2 true positives (regenerated with seed+1). The resolution process is auditable.
- [x] **Time-shift verification as Check 3.** `contamination_check.py` `run_time_shift_check()` enforces three rules: `created_at` ≥ cutoff date, held-out tasks have `bench_snapshot_jittered=True`, `capacity_locked_until` not stale relative to `created_at`. Exposed via `--time-shift` CLI flag.
- [x] **Held-out split uses ±20% jittered bench snapshots.** `methodology.md` documents that held-out tasks use randomized capacity values (seed=99) to prevent exact value memorisation by models trained after the cutoff date.
- [x] **Result is machine-readable and CI-friendly.** `contamination_check.py` exits with code 0 (CLEAN) or 1 (CONTAMINATED), enabling integration into a pre-publish CI gate.

---

## One-Line Disagreement for the Record

Chen et al. recommend dense embedding models for indirect contamination checks. For a domain-specific benchmark where structural overlap (not semantic paraphrase) is the primary risk, TF-IDF cosine is cheaper, faster, and produces fewer false positives on shared domain vocabulary.
