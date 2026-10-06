"""Regression against saved paper draws, plus a holdout-leakage check."""

import unittest

import numpy as np
from threadpoolctl import threadpool_limits

from simulation import SEED, generate, population_reference, whiten, worker


class SimulationTests(unittest.TestCase):
    def test_paper_repetitions(self):
        # Independently archived p=32 results, before the submission extraction.
        accuracies = (
            {"CORR": .6875, "NAIVE": .5625, "ORACLE": .875, "AI-CORR": .5, "LE-CORR": .625},
            {"CORR": .625, "NAIVE": .375, "ORACLE": .6875, "AI-CORR": .625, "LE-CORR": .75},
        )
        maes = (
            {"CORR": .7986732432462155, "NAIVE": 3.1243750393140126, "ORACLE": .24541955638359703},
            {"CORR": .6939562617675864, "NAIVE": 3.1368877827870656, "ORACLE": .14177389787196903},
        )
        with threadpool_limits(limits=1):
            truth, _, xi = population_reference(32)
        self.assertAlmostEqual(truth, .3624233067286206, places=12)
        for repetition in range(2):
            result = worker((32, repetition, truth, xi))
            actual = {row["method"]: row["accuracy"] for row in result["accuracy"]}
            self.assertEqual(actual, accuracies[repetition])
            for row in result["distance"]:
                np.testing.assert_allclose(row["mae"], maes[repetition][row["method"]],
                                           rtol=1e-7, atol=1e-9)
                self.assertEqual(row["pairs"], 32)

    def test_whitening_uses_training_objects_only(self):
        with threadpool_limits(limits=1):
            seed = np.random.SeedSequence([SEED, 32, 0]).spawn(2)[0]
            scms, _, sizes, train, test = generate(32, np.random.default_rng(seed))
            white, n_pool = whiten(scms, sizes, train)
            changed = scms.copy()
            changed[test] *= 10.
            altered, changed_pool = whiten(changed, sizes, train)
        np.testing.assert_array_equal(white[train], altered[train])
        self.assertEqual(n_pool, changed_pool)
        self.assertEqual(n_pool, 5120)


if __name__ == "__main__":
    unittest.main()
