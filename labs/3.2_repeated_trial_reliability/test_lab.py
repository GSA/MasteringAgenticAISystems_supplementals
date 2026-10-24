"""
Chapter 3.2 Lab: self-check.

Run this yourself once you've filled in the TODOs in starter.py:

    python3 test_lab.py

All 16 tests should print "ok". The four TestGuidedFunctions tests pass before
you change anything. Every other test is named after the TODO it checks, so a
FAIL or ERROR line tells you which TODO to revisit. You're done when this file
passes with no edits to itself.

(To check the reference solution instead of your own work, maintainers can run
`LAB_MODULE=solution python3 test_lab.py`. You don't need this as a learner.)
"""

from __future__ import annotations

import importlib
import os
import unittest

import trial_data

MODULE_NAME = os.environ.get("LAB_MODULE", "starter")
mod = importlib.import_module(MODULE_NAME)

DATA = trial_data.generate()
T, F = True, False


def first_trial_only(runs):
    """The same logs, cut down to what a one-run suite would have recorded."""
    return [[trials[0]] for trials in runs]


class TestGuidedFunctions(unittest.TestCase):
    """The provided functions from the We Do section. These pass from the start."""

    def test_flip_rate_on_small_example(self):
        # item 1 never flips; item 2 passes 2 of 4 trials, so 4 of its 6
        # unordered pairs of trials disagree.
        self.assertAlmostEqual(mod.flip_rate([[T, T, T, T], [T, F, T, F]]), (0 + 4 / 6) / 2)

    def test_agent_and_judge_both_flip_on_identical_reruns(self):
        self.assertGreater(mod.flip_rate(DATA.v2), 0.10)
        self.assertGreater(mod.flip_rate(DATA.judge_regrades), 0.05)

    def test_one_run_gate_disagrees_with_itself(self):
        winners = {mod.single_run_winner(DATA.v1, DATA.v2, t) for t in range(len(DATA.v1[0]))}
        self.assertIn("v1", winners)
        self.assertIn("v2", winners)

    def test_simulated_data_is_identical_on_every_machine(self):
        again = trial_data.generate()
        self.assertEqual(again.v2, DATA.v2)
        self.assertEqual(sum(map(sum, DATA.v2)), sum(map(sum, again.v2)))


class TestTodo1PassHatK(unittest.TestCase):
    def test_todo1_known_values(self):
        runs = [[T, T, F, T]]  # 3 passes out of 4
        self.assertAlmostEqual(mod.pass_hat_k(runs, 1), 0.75)
        self.assertAlmostEqual(mod.pass_hat_k(runs, 2), 3 / 6)
        self.assertAlmostEqual(mod.pass_hat_k(runs, 3), 1 / 4)
        self.assertAlmostEqual(mod.pass_hat_k(runs, 4), 0.0)

    def test_todo1_pass_hat_1_is_the_average_success_rate(self):
        mean_rate = sum(map(sum, DATA.v2)) / (len(DATA.v2) * len(DATA.v2[0]))
        self.assertAlmostEqual(mod.pass_hat_k(DATA.v2, 1), mean_rate)

    def test_todo1_pass_hat_k_falls_as_k_grows(self):
        values = [mod.pass_hat_k(DATA.v2, k) for k in range(1, len(DATA.v2[0]) + 1)]
        self.assertEqual(values, sorted(values, reverse=True))
        self.assertLess(values[-1], values[0] - 0.3)

    def test_todo1_rejects_impossible_k(self):
        with self.assertRaises(ValueError):
            mod.pass_hat_k([[T, F]], 0)
        with self.assertRaises(ValueError):
            mod.pass_hat_k([[T, F]], 3)


class TestTodo2PairedBootstrap(unittest.TestCase):
    def test_todo2_identical_versions_give_a_zero_interval(self):
        self.assertEqual(mod.paired_bootstrap_ci(DATA.v1, DATA.v1), (0.0, 0.0))

    def test_todo2_constant_shift_is_recovered_exactly(self):
        v1 = [[F, F, F, F]] * 50
        v2 = [[T, F, F, F]] * 50  # every item is exactly 0.25 better
        lo, hi = mod.paired_bootstrap_ci(v1, v2)
        self.assertAlmostEqual(lo, 0.25)
        self.assertAlmostEqual(hi, 0.25)

    def test_todo2_interval_is_reproducible_and_covers_the_estimate(self):
        first = mod.paired_bootstrap_ci(DATA.v1, DATA.v2, seed=7)
        second = mod.paired_bootstrap_ci(DATA.v1, DATA.v2, seed=7)
        self.assertEqual(first, second)
        rates1 = [sum(t) / len(t) for t in DATA.v1]
        rates2 = [sum(t) / len(t) for t in DATA.v2]
        estimate = sum(b - a for a, b in zip(rates1, rates2)) / len(rates1)
        self.assertLessEqual(first[0], estimate)
        self.assertGreaterEqual(first[1], estimate)

    def test_todo2_one_run_is_inconclusive_but_fifteen_trials_are_not(self):
        lo1, hi1 = mod.paired_bootstrap_ci(first_trial_only(DATA.v1), first_trial_only(DATA.v2))
        lo15, hi15 = mod.paired_bootstrap_ci(DATA.v1, DATA.v2)
        self.assertLess(lo1, 0.0, "a single run should not be able to rule out 'no difference'")
        self.assertGreater(hi1, 0.0)
        self.assertGreater(lo15, 0.0, "15 trials per item should show v2 is better")
        self.assertLess(hi15 - lo15, hi1 - lo1)

    def test_todo2_rejects_mismatched_items(self):
        with self.assertRaises(ValueError):
            mod.paired_bootstrap_ci([[T]], [[T], [F]])


class TestTodo3TrialsNeeded(unittest.TestCase):
    def test_todo3_unanimous_items_need_one_trial(self):
        self.assertEqual(mod.trials_needed([[T, T, T], [F, F, F]]), 1)

    def test_todo3_small_example_needs_three_trials(self):
        # Item 2's first trial disagrees with its 5-trial majority; its first
        # three trials do not. Half the items agree at k=1, all agree at k=3.
        runs = [[T, T, T, T, T], [F, T, T, F, T]]
        self.assertEqual(mod.trials_needed(runs, target=0.95), 3)
        self.assertEqual(mod.trials_needed(runs, target=0.5), 1)

    def test_todo3_simulated_answers(self):
        self.assertEqual(mod.trials_needed(DATA.v2, target=0.95), 5)
        self.assertEqual(mod.trials_needed(DATA.v2, target=0.99), 9)
        with self.assertRaises(ValueError):
            mod.trials_needed([[T, F]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
