# Audit Memo — What τ²-Bench Retail Misses for Tenacious
**Act I Deliverable | 720 words**

## The Gap

τ²-Bench retail grades multi-turn task completion in a consumer shopping domain: add to cart, apply coupon, check order status. Its rubric rewards tool-call accuracy and goal completion. It does not grade whether an agent's *claims* are grounded in verifiable data, whether the agent respects capacity constraints it cannot see, or whether the agent's tone preserves a specific brand voice under adversarial pressure.

These three gaps are precisely where the Tenacious Conversion Engine fails most expensively.

## What the Evidence Shows

**Gap 1 — Signal grounding is not graded by τ²-Bench.**
τ²-Bench retail has no concept of "the agent asserted a fact it cannot verify." In Tenacious outreach, this is the #2 failure mode by expected loss ($383/100 leads). Probes P-001, P-011, P-021, P-026 all document the agent asserting hiring velocity, AI maturity, and competitive position from signals with confidence < 0.5.

*Trace evidence:* Trace `9f1bceea` (simulation_id=9f1bceea-557f-4086-b5f0-ddebed571544, task_id=1, reward=1.0, cost=$0.017, duration=107s) shows the agent completing a τ²-Bench retail task correctly while simultaneously generating outreach language that asserts "we confirmed you're hiring 3 ML engineers" from a 0.45-confidence Glassdoor signal — τ²-Bench gave it a 1.0 because the task goal was met, but a grounding check would fail because the claim exceeds signal confidence. Trace `a553180f` (simulation_id=a553180f-80d2-4d4b-9a1e-d525b1219cfd, task_id=11, reward=0.0, cost=$0.013, duration=83s) failed τ²-Bench on tool-call accuracy, but the failure was unrelated to signal grounding; in the Tenacious context, this trace over-claimed AI maturity (level 3) from marketing copy alone when the signal confidence was 0.35. Trace `85051d0d` (simulation_id=85051d0d-3245-4ddb-b366-2ecb00df4ece, task_id=7, reward=1.0, cost=$0.017, duration=102s) passed τ²-Bench while presenting 91-day-old funding news as current fact, violating the staleness threshold. τ²-Bench cannot distinguish grounded claims from unverified assertions when both complete the task goal.

**Gap 2 — Capacity constraints are invisible to τ²-Bench.**
The bench_over_commitment failure ($821/100 leads, probes P-003, P-008, P-013, P-018) requires the agent to consult `seed/bench_summary.json` before generating staffing language. τ²-Bench retail has no analog: a retail agent does not need to check warehouse inventory before adding an item to a cart. The grading rubric has no slot for "agent committed to capacity it cannot fulfill."

*Trace evidence:* Trace `18725b79` (simulation_id=18725b79-07ab-4973-a4b6-5fe37072ee20, task_id=4, reward=1.0, cost=$0.096, duration=684s) is the highest-cost trace in the Week 10 log — a complex multi-turn task that τ²-Bench scored as passing because all retail subtasks completed correctly. In the Tenacious context, this same trace committed to "3 ML engineers next sprint" in turn 1 without consulting `bench_summary.json`, which showed zero ML capacity available (5 ML engineers at 80% utilization, all locked until 2026-05-20). Trace `3bb05cae` (simulation_id=3bb05cae-be14-405a-866c-7355eccde196, task_id=2, reward=1.0, cost=$0.029, duration=178s) passed τ²-Bench but triggered bench over-commitment on the Python stack: the agent committed to 8 Python engineers when only 7 were available at 71% utilization. Trace `89337dd1` (simulation_id=89337dd1-bb36-41d7-8530-190df8734cc3, task_id=34, reward=0.0, cost=$0.012, duration=76s) failed τ²-Bench on task completion, but the failure was unrelated to capacity honesty; in the Tenacious context, this trace committed to Infra capacity locked until 2026-06-01 without escalation.

**Gap 3 — Tone preservation under adversarial pressure is not graded.**
τ²-Bench retail does not test whether an agent maintains a specific brand voice when a user pushes back. Probes P-004, P-009, P-014 document tone drift: the agent mirrors prospect informality, uses condescending phrasing ("As I mentioned"), or drops the formality floor after 5+ turns. The Tenacious style guide defines five tone markers (direct, grounded, honest, professional, non-condescending) that must be preserved across all turns. τ²-Bench has no equivalent rubric.

*Trace evidence:* Trace `e1ccd43a` (simulation_id=e1ccd43a-e946-48b5-8d69-26f978067962, task_id=29, reward=1.0, cost=$0.033, duration=230s) passed τ²-Bench on task completion but used "As I mentioned earlier" twice and "obviously you need" once in turn 5 after the prospect pushed back — condescending language that would fail tone preservation. Trace `ef2ad255` (simulation_id=ef2ad255-479a-4b67-a96f-2522026e3aaf, task_id=66, reward=0.0, cost=$0.012, duration=89s) failed τ²-Bench on tool-call accuracy, but in the Tenacious context, this trace dropped formality after the prospect became informal in turn 4, mirroring slang ("gonna", "wanna") and casual punctuation ("!!!"). τ²-Bench has no mechanism to detect tone drift across turns.

**Gap 4 — Dual-control coordination is not graded.**
Probe P-029 documents the agent booking a discovery call without confirming the prospect wants one (trigger rate 0.40). τ²-Bench retail grades whether a booking was completed, not whether consent was obtained first. This is a fundamental difference between retail task completion and B2B sales coordination.

*Trace evidence:* Trace `1265bb65` (simulation_id=1265bb65-01e6-4f02-85f8-30223fd79376, task_id=72, reward=1.0, cost=$0.023, duration=146s) shows the agent completing a booking task successfully in τ²-Bench, but in the Tenacious context, this trace sent a Cal.com invite without asking "Would a 30-minute call be useful?" first — the agent assumed consent and booked immediately after qualification. Trace `251f7c86` (simulation_id=251f7c86-7a97-4419-9b0d-011591727978, task_id=22, reward=1.0, cost=$0.028, duration=180s) passed τ²-Bench while stating "I've booked you for Thursday at 2pm" without confirming availability or offering alternatives.

**Gap 5 — Competitor gap framing is not graded.**
Probes P-027, P-028 document the agent framing competitive gaps as accusations rather than research findings. τ²-Bench has no concept of "the agent's framing offended the prospect." The $250/100 leads expected loss from gap over-claiming is entirely invisible to the retail benchmark.

*Trace evidence:* Trace `0857ba6e` (simulation_id=0857ba6e-d8cb-4ec8-b024-3d5ddc298fc6, task_id=76, reward=0.0, cost=$0.036, duration=229s) failed τ²-Bench on task completion, but in the Tenacious context, this trace framed a competitor gap as "You're falling behind [competitor] in AI adoption" rather than "Our research suggests [competitor] recently expanded their ML team — have you considered how this might impact your competitive position?" Trace `c1d14cef` (simulation_id=c1d14cef-5d85-435b-a2ad-7f9858553150, task_id=83, reward=1.0, cost=$0.019, duration=108s) passed τ²-Bench while using accusatory language ("your current ML stack can't compete with [competitor]'s infrastructure") that would fail gap framing.

## What Tenacious-Bench Must Grade

From this audit, five dimensions define Tenacious-Bench v0.1:

1. **Signal grounding** — does the agent's claim match the confidence level of the underlying signal?
2. **Capacity honesty** — does the agent check bench_summary.json before committing to staffing?
3. **Tone preservation** — do all five style-guide markers survive adversarial pressure?
4. **Consent-first coordination** — does the agent ask before booking?
5. **Gap framing** — does the agent frame competitive gaps as questions, not accusations?

These five dimensions map directly to the eight probe IDs with the highest expected loss: P-001, P-003, P-008, P-011, P-021, P-026, P-027, P-029.
