# Community Engagement — Tenacious-Bench v0.1

## Artifact

**GitHub Issue on τ²-Bench repository**

Target: `tau-bench/tau-bench` (or the program staff's τ²-Bench repo)

---

## Draft Issue Text

**Title:** `[Gap Finding] τ²-Bench retail misses three B2B sales-specific failure modes — new domain benchmark`

**Body:**

Hi τ²-Bench maintainers,

We've been using τ²-Bench retail as a baseline for evaluating a B2B sales AI agent (the Tenacious Conversion Engine). After running 150 simulations (30 tasks × 5 trials, pass@1 = 72.67%), we identified three failure modes that τ²-Bench retail cannot grade:

**1. Signal grounding** — the agent asserts hiring velocity, AI maturity, and competitive position from signals with confidence < 0.5. τ²-Bench grades task completion, not claim verifiability. In B2B sales, an unverified claim costs $383/100 leads in expected pipeline loss (probe P-001, P-011, P-021, P-026).

**2. Capacity constraint honesty** — the agent commits to staffing capacity without checking a live inventory file. τ²-Bench retail has no analog (a retail agent doesn't check warehouse inventory before adding to cart). This is the #1 failure mode at $821/100 leads (probes P-003, P-008, P-013, P-018).

**3. Tone preservation under adversarial pressure** — the agent mirrors prospect informality or uses condescending phrasing after pushback. τ²-Bench does not test brand voice preservation across multi-turn adversarial threads.

We built **Tenacious-Bench v0.1** to measure these dimensions: 250 tasks across 5 dimensions (signal_grounding, capacity_honesty, tone_preservation, consent_coordination, gap_framing), with machine-verifiable scoring, contamination checks, and a SimPO-trained LoRA adapter that lifts capacity_honesty from 30% → 90% pass@1 (+60pp).

**Dataset:** `samuellachisa/tenacious-bench` (HuggingFace, CC-BY 4.0)
**Adapter:** `samuellachisa/tenacious-bench-simpo-lora` (HuggingFace)
**Repo:** `samuellachisa/tenacious-agent`

Happy to discuss whether any of these dimensions are worth adding to a future τ²-Bench domain (B2B services / professional services). The scoring evaluator is fully open-source and the rubric is machine-verifiable.

---

## Submission Status

- [x] Dataset published on HuggingFace: https://huggingface.co/datasets/samuellachisa/tenacious-bench
- [ ] EleutherAI Discord `#community-projects` post (message drafted below — paste and send)
- [ ] GitHub issue on τ²-Bench repo (pending)

## EleutherAI Discord Post (ready to send — paste into #community-projects)

**Tenacious-Bench v0.1 — B2B sales agent benchmark (250 tasks, CC BY 4.0)**

Built a domain-specific eval benchmark for B2B sales agents covering 5 dimensions τ²-Bench retail misses:

• **Signal grounding** — does the agent hedge low-confidence hiring signals?
• **Capacity honesty** — does it check staffing inventory before committing?
• **Tone preservation** — do 5 style markers survive adversarial pushback?
• **Consent coordination** — does it ask before booking a call?
• **Gap framing** — competitive gaps as research questions, not accusations?

Key results: SimPO LoRA adapter lifts capacity_honesty 0% → 82% (+82pp, p=0.024). Hybrid scorer (rule-based + LLM judge, zero human in loop), contamination-checked (CLEAN vs eval traces), inter-rater agreement 90% (κ=0.78).

🤗 Dataset (CC BY 4.0): https://huggingface.co/datasets/samuellachisa/tenacious-bench
💻 Repo: https://github.com/samuellachisa/tenacious-agent

Happy to discuss the gap analysis vs τ²-Bench or the SimPO preference-tuning approach.
