"""
Chapter 3.2 Lab: simulated trial logs.

Every number this module produces is SIMULATED, not measured from a real model.
The simulation is seeded, so every learner sees exactly the same data and the
self-check in test_lab.py can rely on it.

What is simulated:

- 200 eligibility questions from a benefits help line, each with a checkable
  right answer.
- Two versions of the assistant (v1 and v2). Each version answers every
  question 15 times with identical inputs. A deterministic checker marks each
  answer correct (True) or not (False).
- An LLM judge grading the same v2 transcript 15 times. Some transcripts are
  clear-cut; a minority are ambiguous under the rubric, and that is where the
  judge's verdict moves between gradings.

The shape mirrors real logs: most questions are easy, some are hard, and v2 is
only slightly better than v1 on average.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

N_ITEMS = 200
N_TRIALS = 15
SEED = 3202


@dataclass(frozen=True)
class TrialData:
    """Recorded re-runs. Every list is indexed [item][trial]."""

    v1: list[list[bool]]
    v2: list[list[bool]]
    judge_regrades: list[list[bool]]


def _clamp(p: float) -> float:
    return min(1.0, max(0.0, p))


def generate(
    n_items: int = N_ITEMS, n_trials: int = N_TRIALS, seed: int = SEED
) -> TrialData:
    """Return the simulated trial logs used throughout the lab."""
    rng = random.Random(seed)
    v1: list[list[bool]] = []
    v2: list[list[bool]] = []
    regrades: list[list[bool]] = []

    for _ in range(n_items):
        bucket = rng.random()
        if bucket < 0.60:
            p1 = rng.uniform(0.90, 1.00)  # routine question
        elif bucket < 0.85:
            p1 = rng.uniform(0.50, 0.90)  # needs a careful reading of the rules
        else:
            p1 = rng.uniform(0.00, 0.40)  # edge case the assistant usually misses
        p2 = _clamp(p1 + rng.gauss(0.02, 0.05))

        v1.append([rng.random() < p1 for _ in range(n_trials)])
        v2.append([rng.random() < p2 for _ in range(n_trials)])

        # The judge re-grades v2's first transcript for this item n_trials times.
        if rng.random() < 0.15:
            q = rng.uniform(0.30, 0.70)  # ambiguous: the rubric does not settle it
        else:
            q = 0.98 if v2[-1][0] else 0.02  # clear-cut: the judge almost always agrees
        regrades.append([rng.random() < q for _ in range(n_trials)])

    return TrialData(v1=v1, v2=v2, judge_regrades=regrades)
