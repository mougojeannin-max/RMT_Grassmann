"""A verification must reject incomplete or merely aggregate-matching results."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from verify import REFERENCE, compare, figure1


class VerificationTests(unittest.TestCase):
    def test_missing_and_duplicate_draws_are_rejected(self):
        reference = pd.DataFrame({"draw": [0, 1], "score": [.25, .75]})
        with self.assertRaises(ValueError):
            compare(reference.iloc[:1], reference, ["draw"], ["score"])
        with self.assertRaises(ValueError):
            compare(pd.concat([reference, reference.iloc[:1]]), reference, ["draw"], ["score"])
        result = compare(reference.iloc[:1], reference, ["draw"], ["score"], partial=True)
        self.assertFalse(result["complete"])
        self.assertEqual(result["missing_rows"], 1)

    def test_equal_means_do_not_hide_different_draws(self):
        reference = pd.DataFrame({"draw": [0, 1], "score": [.25, .75]})
        altered = pd.DataFrame({"draw": [0, 1], "score": [.75, .25]})
        with self.assertRaises(AssertionError):
            compare(altered, reference, ["draw"], ["score"])

    def test_copied_histogram_cannot_hide_a_different_or_nonfinite_spectrum(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / 'figure1_histogram.csv').write_bytes((REFERENCE / 'figure1_histogram.csv').read_bytes())
            values = np.r_[np.ones(254), 4.95, 7.74]
            np.savetxt(folder / 'figure1_eigenvalues.csv', values, delimiter=',', header='eigenvalue', comments='')
            with self.assertRaises(AssertionError):
                figure1(folder)
            values[0] = np.nan
            np.savetxt(folder / 'figure1_eigenvalues.csv', values, delimiter=',', header='eigenvalue', comments='')
            with self.assertRaises(ValueError):
                figure1(folder)


if __name__ == "__main__":
    unittest.main()
