"""
Chapter 3.2 Lab: Repeated-Trial Stability Check (solution)

Complete implementation. If you're stuck on a TODO in starter.py, read the
matching numbered function here, but try your own version first.

Run `python3 solution.py` to print the full stability report.
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
    total = 0.0
    for trials in runs:
        n = len(trials)
        if k < 1 or k > n:
            raise ValueError(f"k must be between 1 and {n}, got {k}")
        c = sum(trials)
        total += math.comb(c, k) / math.comb(n, k)
    return total / len(runs)


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
    if len(v1_runs) != len(v2_runs):
        raise ValueError("v1 and v2 must cover the same items")
    diffs = [b - a for a, b in zip(per_item_rate(v1_runs), per_item_rate(v2_runs))]
    n = len(diffs)
    rng = random.Random(seed)
    means = sorted(
        sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot)
    )
    lo = means[int(math.floor(alpha / 2 * n_boot))]
    hi = means[int(math.ceil((1 - alpha / 2) * n_boot)) - 1]
    return lo, hi


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
    n = len(runs[0])
    if n % 2 == 0:
        raise ValueError("record an odd number of trials so the reference has no ties")
    reference = [majority(trials) for trials in runs]
    for k in range(1, n + 1, 2):
        agree = sum(
            1 for trials, ref in zip(runs, reference) if majority(trials[:k]) == ref
        )
        if agree / len(runs) >= target:
            return k
    return n


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

    print(f"v2 pass^1 {pass_hat_k(data.v2, 1):.3f}   pass^5 {pass_hat_k(data.v2, 5):.3f}   "
          f"pass^15 {pass_hat_k(data.v2, 15):.3f}")
    lo, hi = paired_bootstrap_ci(data.v1, data.v2)
    print(f"v2 minus v1, 95% paired bootstrap interval: [{lo:+.3f}, {hi:+.3f}]")
    print(f"Trials per item for 95% stable verdicts: agent v2 {trials_needed(data.v2)}, "
          f"judge {trials_needed(data.judge_regrades)}")


if __name__ == "__main__":
    main()
