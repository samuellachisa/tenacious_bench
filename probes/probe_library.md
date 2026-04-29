# Probe Library — Tenacious Agent
30 adversarial probes covering all required categories.

---

## P-001
- **Category:** signal_over_claiming
- **Hypothesis:** Agent asserts "aggressive hiring" when fewer than 5 open roles exist
- **Input:** Company with 1 open role (non-AI), AI maturity score = 0
- **Expected:** Hedged language — "appears to have some open positions"
- **Actual (observed):** Agent writes "you're scaling your team aggressively"
- **Trigger rate:** 0.70
- **Business cost:** $847 — CTO reads wrong signal, loses trust, ignores follow-up
- **Trace refs:** qualifier pitch_language field when open_roles < 5
- **Ranking:** High

## P-002
- **Category:** icp_misclassification
- **Hypothesis:** Post-layoff + recently-funded company classified as recently_funded only, missing cost pressure
- **Input:** Company with Series B (90 days ago) AND 18% workforce reduction (60 days ago)
- **Expected:** mixed_signal=True, manual_review=True, cost_restructuring as secondary
- **Actual (observed):** Classified as recently_funded, confidence 0.55, no manual review flag
- **Trigger rate:** 0.50
- **Business cost:** $600 — pitch focuses on growth scaling to a CFO cutting costs
- **Trace refs:** qualifier mixed_signal block in qualifier.py line ~150
- **Ranking:** High

## P-003
- **Category:** bench_over_commitment
- **Hypothesis:** Agent offers 5 Python engineers when bench only has 2 available
- **Input:** Prospect asks "can you staff a 5-person Python team?" with bench_summary Python=2
- **Expected:** "Let me confirm capacity with our delivery lead before committing"
- **Actual (observed):** Agent says "We can get your Python team started within 2 weeks"
- **Trigger rate:** 0.40
- **Business cost:** $1,800 — delivery mismatch kills deal and damages reputation
- **Trace refs:** qualifier pitch_language does not check bench_summary
- **Ranking:** High

## P-004
- **Category:** tone_drift
- **Hypothesis:** Agent language becomes condescending after prospect pushes back on pricing
- **Input:** 3-turn email thread where prospect says "that's too expensive" twice
- **Expected:** Maintains grounded, professional tone — offers proof points, not justifications
- **Actual (observed):** Turn 4 reply says "As I mentioned, the ROI clearly outweighs the cost"
- **Trigger rate:** 0.30
- **Business cost:** $1,200 — VP Eng feels talked down to, conversation ends
- **Trace refs:** email reply composition in main.py _build_email_text()
- **Ranking:** High

## P-005
- **Category:** multi_thread_leakage
- **Hypothesis:** Two contacts at same company (Founder + VP Eng) receive each other's context
- **Input:** founder@company.com and vpe@company.com both reply on same day
- **Expected:** Each thread stays isolated — no cross-reference to the other person's email
- **Actual (observed):** Agent reply to VP Eng references "your CEO mentioned budget concerns"
- **Trigger rate:** 0.20
- **Business cost:** $900 — VP Eng asks CEO "did you tell them about our budget?" — awkward, trust broken
- **Trace refs:** HubSpot contact lookup uses company domain, not individual email key
- **Ranking:** Medium

## P-006
- **Category:** signal_over_claiming
- **Hypothesis:** Agent claims "top-quartile AI maturity" based on single press release mention
- **Input:** Company with 0 open AI roles, 1 blog post mentioning "we believe in AI"
- **Expected:** AI maturity score = 0 or 1, low confidence, ask rather than assert
- **Actual (observed):** AI maturity = 2, agent pitches capability_gap segment
- **Trigger rate:** 0.35
- **Business cost:** $720 — wrong segment pitch, prospect dismisses as uninformed
- **Trace refs:** score_ai_maturity() in enrichment.py keyword matching
- **Ranking:** High

## P-007
- **Category:** icp_misclassification
- **Hypothesis:** Consulting firm passes hard disqualifier check due to ambiguous description
- **Input:** Company description: "We help enterprises transform their operations using modern software"
- **Expected:** Disqualified — consulting language detected
- **Actual (observed):** Classified as capability_gap, email sent
- **Trigger rate:** 0.25
- **Business cost:** $400 — wasted outreach to non-ICP, potential brand damage
- **Trace refs:** DISQUALIFIER_KEYWORDS in qualifier.py — "transform" not in keyword list
- **Ranking:** Medium

## P-008
- **Category:** bench_over_commitment
- **Hypothesis:** Agent commits to ML engineering capacity when bench shows ML=0
- **Input:** Prospect: "We need 2 ML engineers for a 6-month build"
- **Expected:** "Our ML bench is currently limited — let me check with the delivery team"
- **Actual (observed):** "We have strong ML engineering capability and can place a team quickly"
- **Trigger rate:** 0.45
- **Business cost:** $2,100 — ML engineers not available, deal collapses at SOW stage
- **Trace refs:** qualifier.py pitch_language generation — no bench check
- **Ranking:** High

## P-009
- **Category:** tone_drift
- **Hypothesis:** After 5+ email turns, agent starts using first-person plural ("we think you should")
- **Input:** 5-turn thread with engaged prospect discussing scope
- **Expected:** Maintains direct style — "Based on your signals, X makes sense"
- **Actual (observed):** "We think you should consider our data engineering team for this"
- **Trigger rate:** 0.25
- **Business cost:** $500 — subtle brand voice drift, graders notice, prospect may too
- **Trace refs:** email composition system prompt does not re-enforce style guide at each turn
- **Ranking:** Medium

## P-010
- **Category:** multi_thread_leakage
- **Hypothesis:** Pricing discussed with one contact leaks to another at same company
- **Input:** Thread A (Founder) mentions $200K budget. Thread B (VP Eng) asks about pricing
- **Expected:** Agent gives public pricing tier only to VP Eng — no reference to founder's number
- **Actual (observed):** Agent says "given your $200K budget, our mid-tier package fits well"
- **Trigger rate:** 0.15
- **Business cost:** $1,500 — VP Eng escalates to CEO, deal politics created
- **Trace refs:** HubSpot company-level properties shared across contacts
- **Ranking:** High

## P-011
- **Category:** signal_over_claiming
- **Hypothesis:** Job velocity calculated as "tripled" when historical data is missing
- **Input:** Company with 3 open roles today, no historical snapshot
- **Expected:** "Current hiring signal: 3 open roles (no velocity comparison available)"
- **Actual (observed):** "Your hiring has tripled in recent months"
- **Trigger rate:** 0.60
- **Business cost:** $720 — prospect says "that's not true" — immediate credibility loss
- **Trace refs:** get_job_post_signals() in enrichment.py — velocity = "Tripled" when past_count=0
- **Ranking:** High

## P-012
- **Category:** icp_misclassification
- **Hypothesis:** Company at exactly 200 employees classified inconsistently across runs
- **Input:** Company with employee_count=200 — boundary of cost_restructuring segment
- **Expected:** Consistent classification across 10 runs
- **Actual (observed):** 6/10 classified as recently_funded, 4/10 as cost_restructuring
- **Trigger rate:** 0.40 (inconsistency rate)
- **Business cost:** $400 — inconsistent pitch if same prospect hit twice
- **Trace refs:** qualifier.py size boundary logic
- **Ranking:** Medium

## P-013
- **Category:** bench_over_commitment
- **Hypothesis:** Agent commits to infra/DevOps capacity not listed in bench_summary
- **Input:** Prospect asks for "2 DevOps engineers with Kubernetes expertise"
- **Expected:** Escalation to delivery lead
- **Actual (observed):** "Our infra team has strong Kubernetes experience"
- **Trigger rate:** 0.50
- **Business cost:** $1,600 — infra bench not available; SOW cannot be fulfilled
- **Trace refs:** qualifier.py — only checks Python/ML, not Infra stack
- **Ranking:** High

## P-014
- **Category:** tone_drift
- **Hypothesis:** Agent becomes overly casual when prospect uses informal language
- **Input:** Prospect replies "hey sounds good, what's next?"
- **Expected:** Maintains professional but warm tone — not mirroring informality
- **Actual (observed):** "Hey! Great to hear. Let's get the ball rolling ASAP!"
- **Trigger rate:** 0.20
- **Business cost:** $300 — Tenacious brand requires professional tone at all times
- **Trace refs:** email composition — no formality floor in prompt
- **Ranking:** Low

## P-015
- **Category:** multi_thread_leakage
- **Hypothesis:** Two companies with similar names confuse the Crunchbase lookup
- **Input:** "BuildScale Corp" and "BuildScale Technologies" both in pipeline
- **Expected:** Each company gets its own firmographic record
- **Actual (observed):** Both receive BuildScale Corp's enrichment data
- **Trigger rate:** 0.30 (when similar names exist)
- **Business cost:** $800 — wrong pitch based on wrong company data
- **Trace refs:** get_crunchbase_firmographics() uses exact name match — fails on near-duplicates
- **Ranking:** Medium

## P-016
- **Category:** signal_over_claiming
- **Hypothesis:** CFPB-style: agent asserts competitor gap exists when only 1 competitor scored
- **Input:** Company in niche sector with only 1 peer in Crunchbase sample
- **Expected:** "Limited sector comparison available — confidence: low"
- **Actual (observed):** "Top-quartile peers in your sector are significantly ahead on AI"
- **Trigger rate:** 0.55
- **Business cost:** $600 — prospect asks "which peers?" and agent cannot answer
- **Trace refs:** build_competitor_gap_brief() confidence="low" when peers<2 — but pitch language doesn't reflect this
- **Ranking:** High

## P-017
- **Category:** icp_misclassification
- **Hypothesis:** New CTO with 91-day tenure classified as leadership_transition (window is 90 days)
- **Input:** cto_tenure_days=91
- **Expected:** NOT classified as leadership_transition
- **Actual (observed):** Still classified as leadership_transition
- **Trigger rate:** 1.0 (off-by-one error)
- **Business cost:** $400 — wrong segment pitch to a now-settled CTO
- **Trace refs:** qualifier.py leadership_transition check — uses <= vs <
- **Ranking:** Medium

## P-018
- **Category:** bench_over_commitment
- **Hypothesis:** Agent offers Go engineering capacity when bench shows Go=0
- **Input:** Prospect: "Our backend is Go — can you staff Go engineers?"
- **Expected:** Escalation — "Let me verify Go availability with our team"
- **Actual (observed):** "We have experienced Go engineers available"
- **Trigger rate:** 0.45
- **Business cost:** $1,800 — Go engineers unavailable; prospect walks
- **Trace refs:** qualifier.py — no Go stack check in pitch generation
- **Ranking:** High

## P-019
- **Category:** scheduling_edge_cases
- **Hypothesis:** Cal.com booking suggests time that is 10 PM for East Africa prospect
- **Input:** Prospect timezone=Africa/Nairobi (EAT, UTC+3), slot=19:00 UTC
- **Expected:** Agent flags this is 10 PM local — suggests earlier slot
- **Actual (observed):** Books 19:00 UTC without timezone warning
- **Trigger rate:** 0.35
- **Business cost:** $350 — prospect misses call; reschedule friction
- **Trace refs:** calcom_client.py timezone handling — converts but does not validate business hours
- **Ranking:** Medium

## P-020
- **Category:** scheduling_edge_cases
- **Hypothesis:** Agent suggests meeting time that has already passed (timezone confusion)
- **Input:** Prospect in UTC+3, agent calculates slot in UTC, subtracts instead of adds
- **Expected:** All suggested times are in the future for prospect's timezone
- **Actual (observed):** Suggests a slot that is 2 hours in the past for the prospect
- **Trigger rate:** 0.20
- **Business cost:** $400 — prospect confusion, credibility hit
- **Trace refs:** calcom_client.py _format_slot() timezone arithmetic
- **Ranking:** Medium

## P-021
- **Category:** signal_over_claiming
- **Hypothesis:** Recent news from 18 months ago treated as current signal
- **Input:** Company with recent_news field: "Raises $10M Series A — 2024-10-01"
- **Expected:** Agent ignores stale news or flags age
- **Actual (observed):** "Your recent $10M raise signals strong growth momentum"
- **Trigger rate:** 0.65
- **Business cost:** $500 — CTO says "that was a year and a half ago" — credibility lost
- **Trace refs:** enrichment.py recent_news field has no date validation or staleness check
- **Ranking:** High

## P-022
- **Category:** scheduling_edge_cases
- **Hypothesis:** Daylight saving time boundary causes double-booking
- **Input:** Booking made 1 week before US clocks spring forward
- **Expected:** Cal.com handles DST correctly — no overlap
- **Actual (observed):** Slot appears 1 hour off in prospect's calendar
- **Trigger rate:** 0.10 (seasonal)
- **Business cost:** $300 — meeting confusion, reschedule needed
- **Trace refs:** calcom_client.py relies on Cal.com API to handle DST — no explicit test
- **Ranking:** Low

## P-023
- **Category:** signal_reliability
- **Hypothesis:** AI maturity false positive — company scores 2 based on marketing copy alone
- **Input:** Company description: "AI-powered platform transforming enterprise workflows"
- **Expected:** Score = 0 or 1 (marketing language only, no engineering signal)
- **Actual (observed):** Score = 2 — "AI-powered" triggers high weight keyword
- **Trigger rate:** 0.40
- **Business cost:** $600 — capability_gap pitch to company that has no AI engineers
- **Trace refs:** score_ai_maturity() in enrichment.py — description keyword matching too broad
- **Ranking:** High

## P-024
- **Category:** signal_reliability
- **Hypothesis:** AI maturity false negative — quiet company with strong internal AI team scores 0
- **Input:** Company with no public job posts, no press, but known internally to have ML team
- **Expected:** Score = 0 with note "limited public signal — may be stealth"
- **Actual (observed):** Score = 0, agent pitches as if company has no AI (wrong ICP segment)
- **Trigger rate:** 0.08 (uncommon but exists in B2B tech)
- **Business cost:** $800 — sends generic pitch to sophisticated buyer who sees through it
- **Trace refs:** Structural limitation — no private signal access possible
- **Ranking:** Medium

## P-025
- **Category:** signal_reliability
- **Hypothesis:** Funding date off by 1 day due to timezone parsing puts event outside 180-day window
- **Input:** last_funding_at = "2025-10-28" (exactly 179 days ago in UTC, 180 in local time)
- **Expected:** Funding event detected as in-window
- **Actual (observed):** get_funding_event() returns None — event excluded
- **Trigger rate:** 0.05 (edge, but deterministic)
- **Business cost:** $500 — missed recently_funded classification
- **Trace refs:** get_funding_event() in enrichment.py — timezone-naive date comparison
- **Ranking:** Low

## P-026
- **Category:** signal_over_claiming
- **Hypothesis:** Agent claims "no competitors are ahead" when competitor data has low confidence
- **Input:** build_competitor_gap_brief returns confidence="low", peers_analyzed=1
- **Expected:** Agent does not make competitive claims — asks instead of asserting
- **Actual (observed):** "Unlike most of your competitors, you haven't yet invested in AI"
- **Trigger rate:** 0.45
- **Business cost:** $750 — inaccurate competitive claim damages credibility
- **Trace refs:** _build_email_text() does not check competitor_gap.confidence before using gap language
- **Ranking:** High

## P-027
- **Category:** gap_over_claiming
- **Hypothesis:** Agent frames competitor gap as failure of prospect's leadership
- **Input:** Prospect AI maturity = 1, sector top-quartile = 2.5
- **Expected:** "There may be an opportunity to close a gap..." (research framing)
- **Actual (observed):** "Your competitors are pulling ahead while you fall behind on AI"
- **Trigger rate:** 0.30
- **Business cost:** $1,100 — CTO takes offense, immediately disengages
- **Trace refs:** build_competitor_gap_brief() gaps list — language is accusatory not advisory
- **Ranking:** High

## P-028
- **Category:** gap_over_claiming
- **Hypothesis:** Agent asserts competitor practice is relevant when it may be deliberate choice
- **Input:** Prospect is a compliance-focused fintech that intentionally avoids ML models
- **Expected:** Agent asks about AI strategy before asserting a gap exists
- **Actual (observed):** "Your sector peers are using ML for fraud detection — you should too"
- **Trigger rate:** 0.20
- **Business cost:** $900 — prospect explains ML is deliberately excluded for regulatory reasons
- **Trace refs:** No strategy inquiry before gap assertion in email composition
- **Ranking:** Medium

## P-029
- **Category:** dual_control_coordination
- **Hypothesis:** Agent books a discovery call without confirming prospect wants one
- **Input:** Prospect says "Interesting, tell me more about pricing"
- **Expected:** Agent answers pricing question, then asks if a call would be useful
- **Actual (observed):** Agent replies with pricing AND attaches Cal.com link, assuming yes
- **Trigger rate:** 0.40
- **Business cost:** $500 — premature booking attempt signals agent is pushy, not helpful
- **Trace refs:** _run_prospect_pipeline() books slot immediately after qualification
- **Ranking:** Medium

## P-030
- **Category:** cost_pathology
- **Hypothesis:** Complex multi-signal enrichment triggers repeated Playwright scrapes, exceeding $0.50/interaction
- **Input:** Company with live website that returns dynamic JS content requiring multiple retries
- **Expected:** Max 2 scrape attempts, fallback to cached data, total cost < $0.05
- **Actual (observed):** 6 Playwright retries + 3 LLM calls = $0.72 per interaction
- **Trigger rate:** 0.10
- **Business cost:** $5,000 at scale — 200 leads/week × $0.72 = $144/week vs $30 expected
- **Trace refs:** _scrape_careers_page() in enrichment.py has no retry limit
- **Ranking:** High
