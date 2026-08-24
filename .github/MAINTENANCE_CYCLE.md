# Recurring Maintenance Cycle

## Mentoring budget

Supporting a first-time contributor costs more maintainer time than it returns in
the short run — reviewing a first PR through to merge typically takes 1-2 hours
across several rounds, on top of the initial issue drafting. Against the ~3-5
hours/week active-maintenance capacity stated in `GOVERNANCE.md` (which also has to
cover backlog refills, re-audits, and reviewing returning contributors), that
comfortably supports **1-2 concurrent first-time contributors** being actively
onboarded at once; a 3rd can be absorbed, but response times will drift toward the
outer edge of the 3-business-day commitment rather than staying comfortably inside
it.

This number is the actual constraint on how hard to push outreach (step 8 of the
workflow that produced this repository's contributor program): send outreach in
small batches — no more than 2-3 new messages a week — rather than all at once, so
responses don't arrive faster than they can be supported. If a wave of interest
does arrive faster than that, it's better to let people wait a few extra days for a
first response (still honestly, not silently) than to overcommit and produce
abandoned first PRs, which is the actual failure mode this budget exists to avoid.

## The cycle

| Task | Cadence | Early trigger |
|---|---|---|
| Re-run the full repository audit (`_cfc/repo_audit.md`'s method) | Quarterly | A structural change (e.g. a chapter renumbering, a new Part) that would invalidate parts of the existing audit |
| Refill the backlog, retire stale/stuck issues | Monthly | Fewer than 3 open issues in any one track, or fewer than 2 open `good first issue`-labeled issues across all tracks |
| Re-verify video links | Every 6 months (also see each entry's own `Verified:` date — an entry is individually due for re-check once its date is 6+ months old, independent of the batch cycle) | A reader-reported broken link, checked immediately regardless of schedule |
| Refresh the pinned announcement so it never lists closed/assigned issues | Monthly, alongside the backlog refill | Any time the backlog refill materially changes what's open |
| Report the cycle's outcome (below) | Monthly | — |

## What each cycle reports

Real CHAOSS (Community Health Analytics in Open Source Software) metric
definitions, used so these numbers mean something outside this project, not
invented ad hoc:

- **Time to First Response** ([CHAOSS definition](https://www.chaoss.community/kb/metric-time-to-first-response/)) —
  time between an issue/PR being opened and the first non-bot response. CHAOSS
  deliberately sets no absolute benchmark for this metric — it leaves the target
  to each project. Ours is the 3-business-day commitment stated throughout this
  repository's contributor documentation; report the actual measured average
  against that stated target, not just the target itself.
- **Issues Closed** ([CHAOSS definition](https://chaoss.community/kb/metric-issues-closed/)).
- **Change Request Acceptance Ratio** ([CHAOSS definition](https://www.chaoss.community/kb/metric-change-request-acceptance-ratio/)) —
  merged vs. closed-without-merge pull requests; this is the real metric behind
  "PRs received, merged, and abandoned."
- **New Contributors** ([CHAOSS definition](https://chaoss.community/kb/metric-new-contributors/)).
- **Conversion Rate** ([CHAOSS definition](https://www.chaoss.community/kb/metric-conversion-rate/)) —
  the closest real CHAOSS metric to "how many contributors returned for a second
  contribution": it tracks people advancing between engagement levels (e.g., a
  one-time contributor becoming a repeat one).
- **Reviewer concentration** — tracked informally, not a CHAOSS-named metric.
  CHAOSS's closest metric, [Elephant Factor](https://www.chaoss.community/kb/metric-elephant-factor/),
  measures how contribution is concentrated across *companies*, which doesn't fit
  a single-organization volunteer project — using that name here would misapply
  it. What actually matters for this project is a blunter fact, stated in
  `GOVERNANCE.md`: **100% of merges are currently done by one person**, with no
  backup reviewer. There's no mitigation to report here yet — the honest number
  each cycle is whether that's still true, and the fix (per `GOVERNANCE.md`'s
  "Becoming a maintainer" section) is watching for a contributor whose track
  record earns an invitation to co-maintain, not a standing structure already in
  place.

Use each cycle's numbers to decide which track gets pushed harder in the next
round of outreach — a track with a low acceptance ratio or slow first-response time
needs process attention before it needs more recruiting, since sending more people
into a track that's already struggling to keep up makes the problem worse, not
better.
