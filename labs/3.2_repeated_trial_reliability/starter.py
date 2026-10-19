"""
Chapter 3.2 Lab: Repeated-Trial Stability Check (starter)

You are checking whether a benefits help line can trust its offline evaluation
before replacing assistant v1 with v2. Read lab.md first: it explains the
scenario and walks you through the provided functions.

Your job: implement the three functions marked TODO below. Do not change their
names or signatures, because test_lab.py calls them directly.

Run `python3 starter.py` at any point to print the stability report. Parts you
have not implemented yet print as "TODO".
"""

from __future__ import annotations

import math
import random

import trial_data

Runs = list[list[bool]]  # runs[item][trial]


# ---------------------------------------------------------------------------
# Provided (the We Do section walks through these)
# ---------------------------------------------------------------------------


def per_item_rate(runs: Runs) -> list[float]:
    """Fraction of trials that passed, for each item."""
    return [sum(trials) / len(trials) for trials in runs]


def flip_rate(runs: Runs) -> float:
    """Chance that two re-runs of the same item disagree, averaged over items.

    For an item with c passes out of n trials, the share of ordered pairs of
    different trials that disagree is 2c(n - c) / (n(n - 1)). Unlike "did any
    trial disagree", this does not grow just because you recorded more trials.
    """
    total = 0.0
    for trials in runs:
        n = len(trials)
        c = sum(trials)
        total += 2 * c * (n - c) / (n * (n - 1))
    return total / len(runs)


def single_run_success_rate(runs: Runs, trial_index: int) -> float:
    """Success rate you would report if you had run the suite only once."""
    return sum(trials[trial_index] for trials in runs) / len(runs)


def single_run_winner(v1_runs: Runs, v2_runs: Runs, trial_index: int) -> str:
    """The verdict a one-run deploy gate would give: 'v1', 'v2' or 'tie'."""
    a = single_run_success_rate(v1_runs, trial_index)
    b = single_run_success_rate(v2_runs, trial_index)
    if b > a:
        return "v2"
    if a > b:
        return "v1"
    return "tie"


# ---------------------------------------------------------------------------
# TODO 1: pass^k
# ---------------------------------------------------------------------------


def pass_hat_k(runs: Runs, k: int) -> float:
    """Estimate pass^k: the chance that k independent trials of an item all pass,
    averaged over items (Yao et al., tau-bench, arXiv:2406.12045).

    For an item with c passes out of n recorded trials, the unbiased estimate
    is C(c, k) / C(n, k). Raise ValueError if k < 1 or k exceeds the number of
    trials recorded for any item.
    """
    # TODO 1: loop over items. For each, count passes c out of n trials and add
    # math.comb(c, k) / math.comb(n, k). Return the average over items.
    raise NotImplementedError("TODO 1: implement pass_hat_k")


# ---------------------------------------------------------------------------
# TODO 2: paired bootstrap interval on v2 minus v1
# ---------------------------------------------------------------------------


def paired_bootstrap_ci(
    v1_runs: Runs,
    v2_runs: Runs,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for mean(v2 rate - v1 rate).

    Pair the two versions item by item, resample ITEMS with replacement (an
    item keeps both of its rates together), and return the alpha/2 and
    1 - alpha/2 percentiles of the resampled mean difference. Raise ValueError
    if the two versions cover a different number of items.
    """
    # TODO 2: compute one difference per item (v2 rate minus v1 rate), then
    # resample those differences with random.Random(seed). Sort the n_boot
    # resampled means and return the two percentiles described above.
    raise NotImplementedError("TODO 2: implement paired_bootstrap_ci")


# ---------------------------------------------------------------------------
# TODO 3: how many trials before the verdict settles
# ---------------------------------------------------------------------------


def majority(trials: list[bool]) -> bool:
    """True if more than half of the trials passed."""
    return sum(trials) * 2 > len(trials)


def trials_needed(runs: Runs, target: float = 0.95) -> int:
    """Smallest odd k such that, for at least `target` of the items, the
    majority verdict over the first k trials matches the majority verdict over
    all recorded trials. Only odd k are considered, so a majority always exists.
    Raise ValueError if the number of recorded trials is not odd.
    """
    # TODO 3: compute each item's reference verdict with majority(). Then try
    # k = 1, 3, 5, ... and return the first k whose agreement reaches target.
    raise NotImplementedError("TODO 3: implement trials_needed")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def main() -> None:
    data = trial_data.generate()
    n_trials = len(data.v1[0])

    print("Repeated-trial stability report (SIMULATED data, see trial_data.py)")
    print(f"{len(data.v1)} questions x {n_trials} identical re-runs per version\n")

    winners = [single_run_winner(data.v1, data.v2, t) for t in range(n_trials)]
    print("One-run deploy gate, re-run 15 times:")
    print(f"  v2 wins {winners.count('v2')}, v1 wins {winners.count('v1')}, "
          f"ties {winners.count('tie')}")
    print(f"Flip rate, v2 task outcome: {flip_rate(data.v2):.1%}")
    print(f"Flip rate, judge verdict on an identical transcript: "
          f"{flip_rate(data.judge_regrades):.1%}\n")

    try:
        print(f"v2 pass^1 {pass_hat_k(data.v2, 1):.3f}   pass^5 {pass_hat_k(data.v2, 5):.3f}   "
              f"pass^15 {pass_hat_k(data.v2, 15):.3f}")
    except NotImplementedError:
        print("v2 pass^k: TODO 1")
    try:
        lo, hi = paired_bootstrap_ci(data.v1, data.v2)
        print(f"v2 minus v1, 95% paired bootstrap interval: [{lo:+.3f}, {hi:+.3f}]")
    except NotImplementedError:
        print("v2 minus v1, 95% paired bootstrap interval: TODO 2")
    try:
        print(f"Trials per item for 95% stable verdicts: agent v2 {trials_needed(data.v2)}, "
              f"judge {trials_needed(data.judge_regrades)}")
    except NotImplementedError:
        print("Trials per item for 95% stable verdicts: TODO 3")


if __name__ == "__main__":
    main()
