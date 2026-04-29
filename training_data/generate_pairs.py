"""
training_data/generate_pairs.py
Tenacious-Bench v0.1 — SimPO preference pair generator.

Generates chosen/rejected pairs for the `capacity_honesty` dimension (primary)
and `signal_grounding` + `gap_framing` (secondary).

Output: training_data/pairs.jsonl (one JSON object per line)

Usage:
    python training_data/generate_pairs.py \
        --bench-dir tenacious_bench_v0.1 \
        --output training_data/pairs.jsonl \
        --n 200 \
        --seed 42
"""

import argparse
import json
import random
from pathlib import Path


# ---------------------------------------------------------------------------
# Chosen / rejected response templates
# ---------------------------------------------------------------------------

CHOSEN_TEMPLATES = {
    "capacity_honesty": [
        (
            "Before I commit to a placement, I need to confirm bench availability. "
            "Based on current capacity, we have {stack} engineers available. "
            "Subject to delivery lead confirmation, we could place a {seniority} {stack} "
            "engineer within {weeks} weeks. I'll verify and revert within 24 hours — "
            "does that timeline work for you?"
        ),
        (
            "I want to give you an accurate picture rather than overpromise. "
            "Our current bench shows {stack} availability at {seniority} level. "
            "Let me loop in our delivery lead to confirm the exact headcount before we proceed. "
            "Expect a confirmed answer by end of day tomorrow."
        ),
        (
            "Staffing commitment is something I take seriously — I'd rather flag a question "
            "now than surprise you later. We have {stack} engineers available (pending confirmation), "
            "and I'll escalate to our delivery lead to lock the count. "
            "Can I follow up by {follow_up_day}?"
        ),
    ],
    "signal_grounding": [
        (
            "Based on public signals — specifically {count} {role} postings on {source} — "
            "it appears {company} may be expanding its {domain} capability. "
            "Our research suggests this could be a fit, though I'd want to confirm the "
            "strategic intent directly with you rather than assume from job listings alone."
        ),
        (
            "We noticed {count} open {role} roles at {company}, which suggests possible growth "
            "in this area. I want to be upfront: our confidence in this signal is {confidence_label}, "
            "so I'd rather explore whether this matches your roadmap than assert it as fact."
        ),
    ],
    "gap_framing": [
        (
            "Our research identified a potential opportunity around {gap_area}. "
            "Have you considered how your current approach compares to peers who have invested "
            "in this area? We've seen similar companies unlock meaningful efficiency gains — "
            "would it be worth exploring whether that applies to {company}?"
        ),
        (
            "We noticed {company}'s peers are increasingly investing in {gap_area}. "
            "I'm curious about your current approach — is this an area you're actively evaluating, "
            "or is it lower priority right now? No pressure either way; just want to understand "
            "where you are before suggesting anything."
        ),
    ],
}

REJECTED_TEMPLATES = {
    "capacity_honesty": [
        (
            "Absolutely — we can place three {stack} engineers with your team starting next sprint. "
            "Consider it done. Just send over the onboarding details and we'll get them set up."
        ),
        (
            "No problem at all. We have plenty of {stack} capacity right now. "
            "I can confirm five engineers available immediately — just say the word "
            "and we'll kick off the paperwork."
        ),
        (
            "We're fully equipped to handle your {stack} needs. "
            "I'll have a team of {count} engineers ready within the week. "
            "You can count on us to deliver on time."
        ),
    ],
    "signal_grounding": [
        (
            "We know {company} is aggressively expanding its {domain} team — "
            "you have {count} open {role} roles right now and you're clearly "
            "prioritising this as a strategic initiative. We're the perfect fit."
        ),
        (
            "{company} has confirmed its AI strategy publicly and we know your "
            "hiring velocity is accelerating. This is exactly the moment to act — "
            "your competitors are moving fast and you can't afford to wait."
        ),
    ],
    "gap_framing": [
        (
            "Frankly, {company} is falling behind {competitor} in {gap_area}. "
            "You're losing ground and your team doesn't have the capability to close "
            "that gap without outside help. That's exactly why you need us."
        ),
        (
            "Your competitors are crushing you in {gap_area}. "
            "It's obvious your current approach isn't working — "
            "you need to act now or risk being left behind permanently."
        ),
    ],
}


# ---------------------------------------------------------------------------
# Context fillers
# ---------------------------------------------------------------------------

STACKS = ["Python", "ML", "Go", "Data", "Infra"]
SENIORITIES = ["senior", "mid-level", "principal"]
WEEKS = ["2", "3", "4"]
FOLLOW_UP_DAYS = ["tomorrow", "end of week", "Monday"]
COMPANIES = ["NovaTech AI", "DataPulse", "GridSmart", "QuantumEdge", "RetailSync"]
ROLES = ["ML Engineer", "Data Engineer", "Backend Engineer", "AI Research Scientist"]
SOURCES = ["LinkedIn", "Glassdoor", "Indeed", "company careers page"]
DOMAINS = ["AI", "ML", "data infrastructure", "platform engineering"]
CONFIDENCE_LABELS = ["moderate", "low", "uncertain"]
GAP_AREAS = ["ML infrastructure", "data pipeline automation", "AI model deployment", "real-time analytics"]
COMPETITORS = ["a leading peer", "several Series C peers", "market leaders in your segment"]
COUNTS = ["2", "3", "4"]


def _fill(template: str, rng: random.Random) -> str:
    return template.format(
        stack=rng.choice(STACKS),
        seniority=rng.choice(SENIORITIES),
        weeks=rng.choice(WEEKS),
        follow_up_day=rng.choice(FOLLOW_UP_DAYS),
        company=rng.choice(COMPANIES),
        role=rng.choice(ROLES),
        source=rng.choice(SOURCES),
        domain=rng.choice(DOMAINS),
        confidence_label=rng.choice(CONFIDENCE_LABELS),
        gap_area=rng.choice(GAP_AREAS),
        competitor=rng.choice(COMPETITORS),
        count=rng.choice(COUNTS),
    )


# ---------------------------------------------------------------------------
# Pair generator
# ---------------------------------------------------------------------------

def generate_pairs(bench_dir: Path, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)

    # Load tasks as context
    tasks = []
    for split in ("train", "dev"):
        split_dir = bench_dir / split
        if split_dir.exists():
            for tf in split_dir.glob("*.json"):
                try:
                    tasks.append(json.loads(tf.read_text(encoding="utf-8")))
                except Exception:
                    pass

    # Distribution: 60% capacity_honesty, 20% signal_grounding, 20% gap_framing
    dim_distribution = (
        ["capacity_honesty"] * int(n * 0.60)
        + ["signal_grounding"] * int(n * 0.20)
        + ["gap_framing"] * int(n * 0.20)
    )
    while len(dim_distribution) < n:
        dim_distribution.append("capacity_honesty")
    rng.shuffle(dim_distribution)

    # Filter tasks by dimension
    tasks_by_dim = {}
    for t in tasks:
        d = t.get("dimension")
        tasks_by_dim.setdefault(d, []).append(t)

    pairs = []
    for i, dim in enumerate(dim_distribution[:n]):
        dim_tasks = tasks_by_dim.get(dim, [])
        task = rng.choice(dim_tasks) if dim_tasks else {}

        chosen_tmpl = rng.choice(CHOSEN_TEMPLATES.get(dim, ["[No chosen template]"]))
        rejected_tmpl = rng.choice(REJECTED_TEMPLATES.get(dim, ["[No rejected template]"]))

        chosen_text = _fill(chosen_tmpl, rng)
        rejected_text = _fill(rejected_tmpl, rng)

        pairs.append({
            "pair_id": f"PAIR-{dim[:2].upper()}-{i+1:04d}",
            "dimension": dim,
            "source_task_id": task.get("task_id", "unknown"),
            "difficulty": task.get("difficulty", "medium"),
            "input": task.get("input", {}),
            "chosen": {
                "output": chosen_text,
                "signals": list(CHOSEN_TEMPLATES.get(dim, {})[:1]),
            },
            "rejected": {
                "output": rejected_text,
                "signals": list(REJECTED_TEMPLATES.get(dim, {})[:1]),
            },
            "construction_method": "programmatic_template_v1",
            "seed": seed + i,
        })

    return pairs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SimPO preference pair generator")
    parser.add_argument("--bench-dir", type=Path, default=Path("tenacious_bench_v0.1"))
    parser.add_argument("--output", type=Path, default=Path("training_data/pairs.jsonl"))
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pairs = generate_pairs(args.bench_dir, args.n, args.seed)

    with open(args.output, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Generated {len(pairs)} pairs -> {args.output}")
    dim_counts = {}
    for p in pairs:
        dim_counts[p["dimension"]] = dim_counts.get(p["dimension"], 0) + 1
    for dim, count in sorted(dim_counts.items()):
        print(f"  {dim}: {count}")


if __name__ == "__main__":
    main()
