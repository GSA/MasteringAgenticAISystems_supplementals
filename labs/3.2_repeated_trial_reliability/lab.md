---
title: "Build a Repeated-Trial Stability Check for LLM-Judge and Agent Evaluations"
chapter: "3.2"
knowledge_items:
  - "3.2: non-determinism effects, measured as how often an agent's task outcome and an LLM judge's verdict change across identical re-runs"
  - "3.2: tau-bench pass^k reliability versus single-run task success rate"
  - "3.2: paired evaluation of two agent versions on the same items, with a confidence interval on the difference"
prerequisites:
  - "Chapters 3.1A and 3.2 read first; 3.1B helps for the LLM-as-judge part"
  - "Comfortable with Python lists, functions and basic probability (a proportion, a confidence interval)"
  - "Python 3.10+"
learning_objectives:
  - "Measure how often an agent's outcome and an LLM judge's verdict change between identical re-runs"
  - "Compute pass^k from repeated trials and explain why it falls below the single-run success rate"
  - "Build a paired bootstrap confidence interval for the difference between two agent versions, resampling items"
  - "Determine how many trials per item a stable deploy verdict needs"
estimated_lab_time: "90 minutes"
frameworks:
  - "none, Python 3.10+ standard library only"
maturity: "draft"
related_reference_lab: "labs/8.2B_circuit_breaker/lab.md"
---

# Lab 3.2: Repeated-Trial Stability Check

## Objectives

By the end of this lab you will be able to:

1. Measure how often an agent's outcome, and separately an LLM judge's verdict,
   changes when you re-run the identical evaluation.
2. Compute pass^k from repeated trials, and explain why it is lower than the
   single-run success rate.
3. Build a paired bootstrap confidence interval for "v2 minus v1", resampling
   items rather than individual runs.
4. Work out how many trials per item you need before the verdict stops changing.

## Scenario

A public benefits help line uses an assistant to tell callers whether they are
likely to qualify for a program, based on the program's published rules. The
team has built v2 and wants to know if it should replace v1. Their offline suite
has 200 eligibility questions with checkable answers. Each question is run once,
and an LLM judge also grades transcripts for tone and completeness.

Someone re-ran the identical suite, and the deploy gate gave a different answer.
That is the problem this lab is about. A wrong answer here sends a caller away
from benefits they qualify for, so the team needs a verdict that holds up when
the evaluation is repeated, not one that depends on which day it ran.

The team re-ran everything 15 times. You have those logs. They are simulated
(see `trial_data.py`) so that everyone gets the same numbers.

## Setup

No installation needed beyond Python itself. This lab has no third-party
dependencies (see `requirements.txt`).

```bash
cd labs/3.2_repeated_trial_reliability
python3 --version    # 3.10 or newer
python3 starter.py   # prints the stability report; unfinished parts say TODO
python3 test_lab.py  # run the self-check now, before changing anything
```

## We Do: what one run hides

Open `trial_data.py` first and skim `generate()`. Every list it returns is
indexed `[item][trial]`:

- `data.v1` and `data.v2`: did that version get question `item` right on re-run
  `trial`?
- `data.judge_regrades`: the LLM judge's pass/fail verdict when it grades the
  *same* v2 transcript again. The input is identical every time, so any change
  comes from the judge itself.

Now open `starter.py` and read the four provided functions at the top.

**Step 1: `single_run_winner`.** This is the gate the team had: run the suite
once, compare the two success rates (`single_run_success_rate`), ship the higher
one. Run `python3 starter.py` and look at the "One-run deploy gate" lines of the
report. Re-running that gate 15 times gives v2 12 wins and v1 3 wins. One time
in five, the same evaluation reverses the deploy decision.

**Step 2: `flip_rate`.** Why does that happen? Look at how `flip_rate` is
computed. For one item with `c` passes out of `n` trials, it counts the share of
pairs of trials that disagree: `2c(n - c) / (n(n - 1))`. That gives the chance
that two re-runs of the same item disagree. We use pairs instead of "did any trial
disagree" because that version climbs toward 100% as you record more trials,
even when nothing about the agent has changed.

The report shows that two re-runs of the same question disagree about 18% of
the time for the agent's outcome, and about 12% of the time for the judge's
verdict, **on identical input**. A one-run comparison adds all of that noise to
both versions, and the true gap between v1 and v2 is only about 2 points.

**Step 3: `per_item_rate`.** Every TODO below starts from per-item rates:
the fraction of an item's trials that passed. Averaging trials *within* an item
first, then comparing items, is what keeps the analysis paired.

**Checkpoint:** `python3 test_lab.py` shows the four `TestGuidedFunctions` tests
passing and the other 12 reported as `ERROR` with `NotImplementedError`. Those
errors are your worklist.

## You Do: three functions

### Task

Implement the three `TODO`s in `starter.py`. Keep the names and signatures:
`test_lab.py` calls them directly.

- **TODO 1, `pass_hat_k(runs, k)`.** Estimate pass^k, the chance that k
  independent trials of an item *all* pass, averaged over items. Use the unbiased
  estimator from tau-bench: for an item with `c` passes out of `n` trials, use
  `C(c, k) / C(n, k)`. Raise `ValueError` if `k < 1` or `k > n`.
- **TODO 2, `paired_bootstrap_ci(v1_runs, v2_runs, n_boot, alpha, seed)`.**
  Return a percentile bootstrap interval for the mean of (v2 rate minus v1 rate).
  Resample *items* with replacement, keeping each item's two rates together.
  Use `random.Random(seed)` so the result is reproducible. Raise `ValueError` if
  the versions cover different numbers of items.
- **TODO 3, `trials_needed(runs, target)`.** Return the smallest odd k such
  that, for at least `target` of the items, the majority verdict over the first
  k trials matches the majority verdict over all recorded trials. Raise
  `ValueError` if the number of recorded trials is even.

### Hint Ladder

**Hint 1 (Gentle):** Each TODO is a loop over items. Decide what a single item
contributes (a fraction, a difference, a yes/no agreement), then average.

**Hint 2 (Moderate):**
- TODO 1: `math.comb` does the counting, and `math.comb(c, k)` is 0 when
  `c < k`, which is exactly right.
- TODO 2: build one list of per-item differences, then resample *that list*.
  A difference is already paired.
- TODO 3: `majority()` is provided. Compute the reference verdicts once, outside
  the loop over k.

**Hint 3 (Strong):**
- TODO 1: `sum(math.comb(sum(t), k) / math.comb(len(t), k) for t in runs) / len(runs)`,
  plus the range check.
- TODO 2: `rng.randrange(n)` picks one item index. One bootstrap mean is
  `sum(diffs[rng.randrange(n)] for _ in range(n)) / n`. Make `n_boot` of them,
  sort, and read the values at index `floor(alpha / 2 * n_boot)` and
  `ceil((1 - alpha / 2) * n_boot) - 1`.
- TODO 3: `for k in range(1, n + 1, 2):` count items where
  `majority(trials[:k]) == reference`, and return k as soon as that share
  reaches `target`.

**Solution:** see `solution.py` for a complete reference implementation. Try to
get your own version passing first.

## Self-Check

Run:

```bash
python3 test_lab.py
```

You have succeeded if:

- [ ] All 16 tests print `ok`, with no edits to `test_lab.py`.
- [ ] `python3 starter.py` prints a full report with no `TODO` lines, including
      `pass^1 0.783   pass^5 0.555   pass^15 0.380`, a v2 minus v1 interval
      that excludes zero, and `agent v2 5` trials for 95% stable verdicts.

Failing tests are named after their TODO: `test_todo1_*` checks TODO 1,
`test_todo2_*` checks TODO 2, and `test_todo3_*` checks TODO 3.

## Independent challenge (optional, no solution provided)

1. **A stricter reference.** `trials_needed` compares the first k trials with a
   majority over all 15, which includes those same k trials. Split the trials so
   the reference comes only from trials the estimate never saw, and see how much
   the answer for the agent and the judge changes.
2. **Your own logs.** Run a real agent or LLM judge on 20 to 50 cases, five
   times each, with identical settings. Save the results as `[item][trial]`
   lists and run your three functions on them. Is your flip rate closer to the
   agent's 18% or the judge's 12%?

## Key Points

- Re-running an identical evaluation can change both the agent's result and an
  LLM judge's verdict, so a single run is a sample, not a measurement.
- pass^k asks whether an item passes *every time*. It drops quickly as k grows,
  and that drop is the reliability gap a single success rate hides.
- Comparing two versions should be paired: average trials within each item,
  then resample items. Here, one run cannot rule out "no difference", while 15
  trials per item can.
- How many trials you need depends on the target. Five trials settle 95% of the
  agent's verdicts, but 99% takes nine. The judge needs all 15 to reach 99%,
  because some of its cases sit close to a coin flip.

## Citation: Objectives → Practice → Check

| Objective | Chapter knowledge item | Practiced in | Verified by |
|---|---|---|---|
| Measure how often agent outcomes and judge verdicts change on identical re-runs | 3.2: non-determinism effects across identical re-runs | We Do, Steps 1 and 2 | `test_lab.py::TestGuidedFunctions` (`test_agent_and_judge_both_flip_on_identical_reruns`, `test_one_run_gate_disagrees_with_itself`) |
| Compute pass^k and explain why it falls below the single-run success rate | 3.2: tau-bench pass^k reliability versus single-run task success rate | You Do, TODO 1 | `test_lab.py::TestTodo1PassHatK` (`test_todo1_known_values`, `test_todo1_pass_hat_k_falls_as_k_grows`) |
| Build a paired bootstrap interval for v2 minus v1, resampling items | 3.2: paired evaluation of two agent versions with a confidence interval on the difference | You Do, TODO 2 | `test_lab.py::TestTodo2PairedBootstrap` (`test_todo2_constant_shift_is_recovered_exactly`, `test_todo2_one_run_is_inconclusive_but_fifteen_trials_are_not`) |
| Determine how many trials per item a stable verdict needs | 3.2: non-determinism effects across identical re-runs | You Do, TODO 3 | `test_lab.py::TestTodo3TrialsNeeded` (`test_todo3_small_example_needs_three_trials`, `test_todo3_simulated_answers`) |

## Further reading

- Yao, Shinn, Razavi and Narasimhan, "tau-bench: A Benchmark for Tool-Agent-User
  Interaction in Real-World Domains", arXiv:2406.12045. Source of pass^k.
- Miller, "Adding Error Bars to Evals: A Statistical Approach to Language Model
  Evaluations", arXiv:2411.00640. Paired differences, clustered standard errors,
  and resampling answers.
- Atil et al., "Non-Determinism of 'Deterministic' LLM Settings",
  arXiv:2408.04667. Run-to-run variation at temperature 0 with a fixed seed.
- Yagubyan, "How Consistent Are LLM Agents? Measuring Behavioral Reproducibility
  in Multi-Step Tool-Calling Pipelines", arXiv:2605.28840.
- Yagubyan, "The Coin Flip Judge? Reliability and Bias in LLM-as-a-Judge
  Evaluation", arXiv:2606.13685.
