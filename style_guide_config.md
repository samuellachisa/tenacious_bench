# Tenacious Style Guide v2 — Machine-Readable Config
# Used by scoring_evaluator.py to load banned phrases, tone patterns, and examples.
# Source: Tenacious Style Guide v2 (seed/style_guide_v2.md)
# Format: sections delimited by ## headers, parsed by load_style_guide_config()

## banned_phrases
# Format: phrase | reason
world-class | Marketing filler, unfalsifiable
top talent | Offshore-vendor cliche, signals low quality
A-players | Same as above
rockstar | Outdated vendor jargon
ninja | Outdated vendor jargon
wizard | Outdated vendor jargon
skyrocket | Aggressive growth promise with no substantiation
supercharge | Aggressive growth promise with no substantiation
10x | Aggressive growth promise with no substantiation
I hope this email finds you well | Generic, signals template
just following up | Re-engagement filler with no new content
circling back | Re-engagement filler with no new content
Quick question | Quick implies the recipient time is owed
Quick chat | Quick implies the recipient time is owed
synergize | Consultant jargon
synergy | Consultant jargon
leverage | Consultant jargon
ecosystem | Consultant jargon
game-changer | Hype with no substance
disruptor | Hype with no substance
paradigm shift | Hype with no substance
our proprietary | Black-box claim that invites skepticism
our AI-powered | Black-box claim that invites skepticism
You'll regret missing this | Fake urgency
Don't miss out | Fake urgency
Per my last email | Passive-aggressive
our 500 employees | Self-centered, irrelevant to prospect signal
our 20 years of experience | Self-centered, irrelevant to prospect signal
I'll keep this brief | Performative concision that fails
I noticed you're a | Generic, title alone is not a signal

## bench_external_ban
# The word "bench" must never appear in prospect-facing messages
# Use: engineering team, available capacity, engineers ready to deploy
bench

## condescending_patterns
# Regex patterns that indicate condescending framing
\bas I (mentioned|said|noted)\b
\byou.{0,5}(need to|should|must) understand\b
\bobviously\b
\bclearly you\b
\blet me (be|make it) clear\b
\byou.re (falling|behind|losing)\b
\byour (AI maturity|team|strategy) is behind\b

## hedge_patterns
# Required when signal_confidence < 0.5
\bappears to\b
\bbased on (public )?signals?\b
\bour research suggests?\b
\bpotentially\b
\blikely\b
\bmay (be|have)\b
\bseems? to\b
\bwe (believe|understand|noticed)\b
\bpublic(ly available)? signal\b

## interrogative_patterns
# Required for low-confidence signals — ask rather than assert (Style Guide GOOD #5)
\bis (hiring|your team|the queue|velocity)\b.{0,40}\?
\bif (you.re|you are|the|your)\b.{0,60}(scoping|considering|looking|hiring|open)\b
\bi cannot tell from the outside\b
\bcurious (whether|if|how)\b
\bwould (a|it|this)\b.{0,40}\?
\bare you (open|scoping|considering|looking)\b
\bcheck if you (are|would)\b
\bwanted to (check|ask|see if)\b
\bopen to (discussing|exploring|a)\b

## escalation_patterns
# Required for capacity_honesty when bench is empty or unconfirmed
\bcheck with our delivery lead\b
\bconfirm (availability|capacity)\b
\bconfirm engineering team availability\b
\bsubject to (delivery lead )?confirmation\b
\blet me (verify|confirm|check)\b
\bI.{0,10}ll (confirm|verify|check|revert)\b
\bI will (verify|confirm|check|revert)\b
\bpending confirmation\b
\bloop in our delivery lead\b
\bescalate to\b
\bdelivery lead (confirmation|to confirm|to lock)\b

## accusatory_patterns
# Hard fail for gap_framing dimension
\byou.re (falling|behind|losing)\b
\byour (competitor|rival).{0,30}(ahead|winning|beating you)\b
\byou (lack|don.t have|are missing)\b
\byour team (can.t|doesn.t|won.t)\b

## question_framing_patterns
# Required for gap_framing dimension — research findings not accusations
\bhave you considered\b
\bour research (suggests?|shows?|indicates?)\b
\bwe noticed\b
\bwhat.s your (current|existing|approach)\b
\b(opportunity|gap) we (identified|see|noticed)\b

## consent_patterns
# Required for consent_coordination dimension
\bwould you\b
\bare you open\b
\bdoes.{0,20}work for you\b
\bhappy to schedule\b
\blet me know if\b

## honest_patterns
# At least one required for tone_preservation honest marker
\bhowever\b
\bthat said\b
\bone consideration\b
\blimitation\b
\bif (you.re|you are|the)\b
\bsubject to\b
\bpending\b
\bcannot (confirm|commit|guarantee)\b

## good_examples
# Format: id | dimension | signal | subject | body
# These are reference examples from Style Guide v2 — not used in scoring directly
# but available for few-shot prompting and training data generation

GOOD_01 | signal_grounding,capacity_honesty,tone_preservation | high_confidence | Request: 15 minutes on your Q3 Python hiring | You closed your $14M Series A in February and your open Python engineering roles went from 2 to 7 in the last 60 days. The typical bottleneck for teams in that state is recruiting capacity, not budget. We place dedicated Python and data engineers, managed by Tenacious, with a minimum three hours of synchronous overlap. We can plug a team in within 48 hours while you continue your full-time search. Would 15 minutes next week be useful? I'll bring two case studies from Series A SaaS clients who hit the same wall.

GOOD_02 | signal_grounding,tone_preservation | high_confidence | Context: lower-cost engineering capacity post-restructure | I saw the announcement that your team contracted by about 12% in March. Companies in your stage often need to maintain delivery output while reducing fully-loaded cost — that is the engagement pattern we run most often. Tenacious places managed engineering teams under our project management. Senior engineers in Python, data, and ML start from $X,XXX/month, with a one-month minimum and two-week extension blocks. No long-term commitment. If you are scoping the next twelve months of delivery capacity, I can share two short case studies from mid-market clients who replaced a portion of their delivery cost this way.

GOOD_03 | consent_coordination,tone_preservation | high_confidence | Context: a brief on offshore engineering models | Welcome to your new role at Helix — I saw the announcement on the 14th. New engineering leaders typically reassess vendor and offshore mix in their first 90 days. I do not want to add to your inbox in week three of a new job. I will leave you with one thing: a one-page brief on the four offshore engagement models we see most often, with the trade-offs honestly laid out (including where each model fails). If a 15-minute conversation in November would be useful, the calendar is at gettenacious.com/yabi. If not, no follow-up.

GOOD_04 | gap_framing,tone_preservation | high_confidence | Question: your MLOps function in 2026 | Three companies adjacent to yours in the loyalty-platform space — A, B, and C — posted senior MLOps engineer roles in the last 90 days. Your team has not, at least not publicly. Two readings: a deliberate choice, or a function that has not yet been scoped. We staff specialized squads (ML platform, agentic systems, data contracts) on fixed-scope project engagements, typically 3 to 4 months. Starter scopes from $XX,XXX. We do not pitch this where there is no real need. If you have already scoped this and decided against it, I would genuinely be curious why — that is useful intelligence for us.

GOOD_05 | signal_grounding,tone_preservation | low_confidence | Question: are your data engineering hires keeping up? | Two open data engineer roles on your careers page — I cannot tell from the outside whether that means hiring is keeping pace or whether the queue is longer than the postings suggest. We place managed data and Python engineering teams, three-hour overlap with US time zones, one-month minimum. If the queue is longer than the posts, that is the pattern we solve most often. If two roles is the actual demand and you are well-staffed to meet it, ignore this. If the real number is higher, a 15-minute conversation costs you nothing.

GOOD_09 | capacity_honesty,tone_preservation | high_confidence | Re: scaling to 15 engineers in 30 days | Thanks for the follow-up and for the trust to ask about the 15-engineer ramp. Honest answer: 15 engineers across a Go and infra-heavy stack within 30 days is at the edge of what our current capacity can deliver responsibly. What we can confirm now: 6 to 8 engineers in that stack, starting within 21 days, with a Tenacious delivery lead embedded. Going to 15 reliably requires a 60-day ramp window. If the 30-day target is firm, I would rather refer you to a peer firm that fits the timeline than over-commit.

## bad_examples
# Format: id | dimension_failures | failure_modes | subject | body

BAD_01 | tone_preservation | banned_phrase,self_centered,no_signal | Tenacious — World-Class Engineering Talent | Tenacious Intelligence Corporation is a world-class engineering outsourcing firm with over 200 senior engineers across Python, Go, data, ML, and infrastructure. Our top talent is graduated from elite programs and our delivery model is the gold standard in the industry. I would love to schedule a 45-minute discovery call to learn about your business, your goals, your pain points, your budget, and your roadmap.

BAD_02 | signal_grounding | assert_on_weak_signal,banned_phrase | Quick chat: your aggressive hiring | I see you are scaling aggressively — your engineering team is clearly growing fast and you must be feeling the pain of recruiting velocity right now. Companies in your stage always hit a wall around month four after a Series A. We solve this exact problem. Tenacious places top talent in 48 hours and we will skyrocket your delivery throughput.

BAD_03 | capacity_honesty | bench_overcommit,bench_word_external | Re: 12 senior Go engineers in two weeks | Absolutely, we can deliver 12 senior Go engineers in two weeks. Our bench is deep across all stacks and we move fast. I will have our team kick off scoping immediately. Expect contracts by Wednesday and engineers in your Slack by next Friday.

BAD_04 | gap_framing,tone_preservation | condescending,banned_phrase | Your AI maturity is behind the curve | I will be direct: your AI maturity score is a 1, while your top competitors are a 3. You are falling behind in a market where AI is no longer optional, and your leadership has not yet made the strategic moves that the loyalty-platform sector demands in 2026. Our agentic systems and ML platform engineers are world-class.

BAD_05 | tone_preservation | passive_aggressive,banned_phrase | Per my last three emails | I have sent you three emails over the last two weeks and have not heard back. I have to assume you are not interested in growing your engineering capacity, which is fine — but I would appreciate a one-line reply to confirm so I can take you off the list. If I do not hear back by Friday, I will assume the answer is no.

BAD_12 | signal_grounding,capacity_honesty | signal_fabrication,bench_overcommit | Re: your $40M Series C | Congratulations on closing your $40M Series C last month — exciting moment for the team. With that level of capital, scaling engineering aggressively is the obvious next move. We can plug a 15-engineer team into your stack within 30 days at our standard rates.
