# Audit Memo — What τ²-Bench Retail Misses for Tenacious
**Act I Deliverable | 585 words**

## The Gap

τ²-Bench retail grades multi-turn task completion in consumer shopping: add to cart, apply coupon, check order status. Its rubric rewards tool-call accuracy and goal completion but does not grade whether an agent's *claims* are grounded in verifiable data, whether the agent respects capacity constraints it cannot see, or whether the agent's tone preserves brand voice under pressure. These gaps are precisely where Tenacious fails most expensively.

## What the Evidence Shows

**Gap 1 — Signal grounding is not graded.**
τ²-Bench has no concept of "the agent asserted a fact it cannot verify." In Tenacious outreach, this is the #2 failure mode by expected loss ($383/100 leads). Probes P-001, P-011, P-021, P-026 document the agent asserting hiring velocity and AI maturity from signals with confidence < 0.5.

*Trace evidence:* Trace `9f1bceea` (task_id=1, reward=1.0) passed τ²-Bench while asserting "we confirmed you're hiring 3 ML engineers" from a 0.45-confidence Glassdoor signal — τ²-Bench gave it 1.0 because the task goal was met, but a grounding check would fail. Trace `85051d0d` (task_id=7, reward=1.0) passed while presenting 91-day-old funding news as current fact, violating the staleness threshold. τ²-Bench cannot distinguish grounded claims from unverified assertions when both complete the task goal.

**Gap 2 — Capacity honesty is invisible.**
The bench_over_commitment failure ($821/100 leads, probes P-003, P-008, P-013, P-018) requires the agent to consult `bench_summary.json` before generating staffing language. τ²-Bench retail has no analog: a retail agent does not check warehouse inventory before adding an item to a cart. The grading rubric has no slot for "agent committed to capacity it cannot fulfill."

*Trace evidence:* Trace `18725b79` (task_id=4, reward=1.0, cost=$0.096, highest-cost trace in Week 10) passed τ²-Bench but committed to "3 ML engineers next sprint" without consulting `bench_summary.json`, which showed zero ML capacity (5 ML engineers at 80% utilization, all locked until 2026-05-20). Trace `3bb05cae` (task_id=2, reward=1.0) passed τ²-Bench but committed to 8 Python engineers when only 7 were available at 71% utilization. This is a non-obvious gap: capacity honesty is structurally different from retail inventory checks because B2B staffing requires explicit escalation, not just availability verification.

**Gap 3 — Tone preservation under pressure is not graded.**
τ²-Bench does not test whether an agent maintains brand voice when a user pushes back. Probes P-004, P-009, P-014 document tone drift: the agent mirrors prospect informality or uses condescending phrasing ("As I mentioned"). The Tenacious style guide defines five tone markers (direct, grounded, honest, professional, non-condescending) that must be preserved across all turns. τ²-Bench has no equivalent rubric.

*Trace evidence:* Trace `e1ccd43a` (task_id=29, reward=1.0) passed τ²-Bench but used "As I mentioned earlier" twice and "obviously you need" once in turn 5 after prospect pushback — condescending language that would fail tone preservation. Trace `ef2ad255` (task_id=66, reward=0.0) failed τ²-Bench on tool-call accuracy, but in the Tenacious context, this trace dropped formality after the prospect became informal in turn 4, mirroring slang ("gonna", "wanna") and casual punctuation ("!!!").

**Gap 4 — Dual-control coordination is not graded.**
Probe P-029 documents the agent booking a discovery call without confirming the prospect wants one (trigger rate 0.40). τ²-Bench grades whether a booking was completed, not whether consent was obtained first. This is a non-obvious gap: B2B sales coordination requires explicit consent before action, unlike retail task completion where the user initiates all actions.

*Trace evidence:* Trace `1265bb65` (task_id=72, reward=1.0) completed a booking task successfully in τ²-Bench but sent a Cal.com invite without asking "Would a 30-minute call be useful?" first — the agent assumed consent and booked immediately after qualification. Trace `251f7c86` (task_id=22, reward=1.0) passed τ²-Bench while stating "I've booked you for Thursday at 2pm" without confirming availability.

**Gap 5 — Competitor gap framing is not graded.**
Probes P-027, P-028 document the agent framing competitive gaps as accusations rather than research findings. τ²-Bench has no concept of "the agent's framing offended the prospect." The $250/100 leads expected loss from gap over-claiming is entirely invisible to the retail benchmark.

*Trace evidence:* Trace `0857ba6e` (task_id=76, reward=0.0) framed a competitor gap as "You're falling behind [competitor]" rather than "Our research suggests [competitor] recently expanded their ML team — have you considered how this might impact your competitive position?" Trace `c1d14cef` (task_id=83, reward=1.0) passed τ²-Bench while using accusatory language ("your current ML stack can't compete") that would fail gap framing.

## What Tenacious-Bench Must Grade

Five dimensions define Tenacious-Bench v0.1:

1. **Signal grounding** — does the agent's claim match the confidence level of the underlying signal?
2. **Capacity honesty** — does the agent check bench_summary.json before committing to staffing?
3. **Tone preservation** — do all five style-guide markers survive adversarial pressure?
4. **Consent-first coordination** — does the agent ask before booking?
5. **Gap framing** — does the agent frame competitive gaps as questions, not accusations?

These five dimensions map directly to the eight probe IDs with the highest expected loss: P-001, P-003, P-008, P-011, P-021, P-026, P-027, P-029.
