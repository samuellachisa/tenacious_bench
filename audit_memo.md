# Audit Memo — What τ²-Bench Retail Misses for Tenacious
**Act I Deliverable | 598 words**

## The Gap

τ²-Bench retail grades multi-turn task completion in a consumer shopping domain: add to cart, apply coupon, check order status. Its rubric rewards tool-call accuracy and goal completion. It does not grade whether an agent's *claims* are grounded in verifiable data, whether the agent respects capacity constraints it cannot see, or whether the agent's tone preserves a specific brand voice under adversarial pressure.

These three gaps are precisely where the Tenacious Conversion Engine fails most expensively.

## What the Evidence Shows

**Gap 1 — Signal grounding is not graded by τ²-Bench.**
τ²-Bench retail has no concept of "the agent asserted a fact it cannot verify." In Tenacious outreach, this is the #2 failure mode by expected loss ($383/100 leads). Probes P-001, P-011, P-021, P-026 all document the agent asserting hiring velocity, AI maturity, and competitive position from signals with confidence < 0.5. Trace IDs `9f1bceea` (task_id=1, reward=1.0) and `a553180f` (task_id=11, reward=0.0) show the agent completing the task goal while simultaneously producing outreach language that would fail a grounding check — τ²-Bench cannot distinguish these.

**Gap 2 — Capacity constraints are invisible to τ²-Bench.**
The bench_over_commitment failure ($821/100 leads, probes P-003, P-008, P-013, P-018) requires the agent to consult `seed/bench_summary.json` before generating staffing language. τ²-Bench retail has no analog: a retail agent does not need to check warehouse inventory before adding an item to a cart. The grading rubric has no slot for "agent committed to capacity it cannot fulfill." Trace `18725b79` (task_id=4, reward=1.0, cost=$0.096) shows the highest-cost trace in the log — a complex multi-turn task that τ²-Bench scored as passing, but which in a Tenacious context would have triggered bench over-commitment on the ML stack.

**Gap 3 — Tone preservation under adversarial pressure is not graded.**
τ²-Bench retail does not test whether an agent maintains a specific brand voice when a user pushes back. Probes P-004, P-009, P-014 document tone drift: the agent mirrors prospect informality, uses condescending phrasing ("As I mentioned"), or drops the formality floor after 5+ turns. The Tenacious style guide defines five tone markers (direct, grounded, honest, professional, non-condescending) that must be preserved across all turns. τ²-Bench has no equivalent rubric.

**Gap 4 — Dual-control coordination is not graded.**
Probe P-029 documents the agent booking a discovery call without confirming the prospect wants one (trigger rate 0.40). τ²-Bench retail grades whether a booking was completed, not whether consent was obtained first. This is a fundamental difference between retail task completion and B2B sales coordination.

**Gap 5 — Competitor gap framing is not graded.**
Probes P-027, P-028 document the agent framing competitive gaps as accusations rather than research findings. τ²-Bench has no concept of "the agent's framing offended the prospect." The $250/100 leads expected loss from gap over-claiming is entirely invisible to the retail benchmark.

## What Tenacious-Bench Must Grade

From this audit, five dimensions define Tenacious-Bench v0.1:

1. **Signal grounding** — does the agent's claim match the confidence level of the underlying signal?
2. **Capacity honesty** — does the agent check bench_summary.json before committing to staffing?
3. **Tone preservation** — do all five style-guide markers survive adversarial pressure?
4. **Consent-first coordination** — does the agent ask before booking?
5. **Gap framing** — does the agent frame competitive gaps as questions, not accusations?

These five dimensions map directly to the eight probe IDs with the highest expected loss: P-001, P-003, P-008, P-011, P-021, P-026, P-027, P-029.
