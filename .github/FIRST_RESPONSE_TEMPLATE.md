# First-response template

The first reply to a first-time contributor decides whether they come back —
research on this project cited in `_cfc/workstreams.md` §5 is why the 3-business-day
commitment exists at all. This template exists so that commitment is easy to keep
under real time pressure. Three rules, then the template:

1. **Acknowledge the work specifically**, not generically — name what they actually
   did, not "thanks for your PR."
2. **Be concrete about what happens next** — a date, a specific next step, or a
   specific question, never "we'll take a look."
3. **Separate blocking changes from suggestions explicitly.** A contributor who
   can't tell which of your comments they *must* address before merge, versus
   which are optional polish, will over-fix or under-fix. Label them.

## Template — new issue claimed

> Thanks for picking this up — assigning it to you now. [One sentence on anything
> issue-specific worth flagging up front, or delete this line if there's nothing to
> add.] Ping this issue if anything in the definition of done is unclear once you
> get into it; that's normal, not a sign you're doing it wrong.

## Template — first PR from a new contributor, needs changes

> Thanks for this — [name the specific thing they built, e.g. "the HALF_OPEN
> transition logic is solid, and I like that you added a test for the concurrent-
> trial case beyond what the issue asked for"].
>
> **Blocking, before this can merge:**
> - [Specific, actionable item]
> - [Specific, actionable item]
>
> **Optional, your call:**
> - [Suggestion — style, an edge case worth considering, etc.]
>
> Once the blocking items are addressed this should be quick to re-review. Let me
> know if anything above isn't clear.

## Template — first PR from a new contributor, ready to merge

> This looks good — merging. [One specific thing you appreciated, not generic
> praise.] Thanks for the contribution, and for [the specific track] more broadly —
> if you're interested in another one, `track:<label>` issues are open, or feel
> free to just say what you'd like to work on next.

## Template — PR needs to be split (oversized)

> This is good work, but it's larger than our roughly-20-item pull request
> guideline, which is about keeping review fast on a small team, not about your
> PR's quality. Could you split it along [scaffolding / guided section /
> solution+check / polish, or whichever split fits] per `CONTRIBUTING.md`'s
> splitting guidance? Happy to help think through where the natural seams are if
> that's useful.

## Template — content-review finding, verdict is "major revisions"

> Thanks for the detailed review — [one specific thing about the finding that was
> genuinely useful]. Filing this as "major revisions": [one-sentence summary of
> why]. Leaving the issue open with your findings attached — anyone (including you,
> if you want it) can pick it up from here. No action needed from you unless you'd
> like to take the fix yourself.
