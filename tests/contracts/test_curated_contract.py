"""Exercise the zero-argument loader through eight checksum-pinned fixtures."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from data import curate
from tests.fixtures.curation import write_release


class CuratedContractTests(unittest.TestCase):
    def test_zero_argument_load_curated_contract_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = write_release(root)
            with patch.object(curate, "DEFAULT_OUTPUT", root), \
                    patch.object(curate, "load_manifest", return_value=manifest):
                a = curate.load_curated()
                b = curate.load_curated()
        self.assertIsInstance(a, pd.DataFrame)
        assert_frame_equal(a, b)
        self.assertEqual(list(a.columns), [
            "id", "gps", "ifo", "label", "confidence", "duration", "peak_frequency",
            "central_freq", "bandwidth", "amplitude", "snr", "q_value", "run", "is_held_out",
        ])
        self.assertGreater(len(a), 0)
        self.assertTrue(a.id.is_unique)
        self.assertFalse(a.isna().any().any())
        self.assertEqual(set(a.ifo), {"H1", "L1"})
        self.assertEqual(set(a.label), {*curate.KNOWN_CLASSES, curate.HELD_OUT})
        self.assertTrue(a.confidence.between(.9, 1.).all())
        for column in ("gps", "confidence", *curate.FEATURE_COLUMNS):
            self.assertEqual(a[column].dtype, np.dtype("float64"))
            self.assertTrue(np.isfinite(a[column]).all())
        self.assertEqual(a[list(curate.FEATURE_COLUMNS)].shape, (len(a), 7))
        self.assertEqual(a.is_held_out.dtype, np.dtype("bool"))
        self.assertTrue(a.loc[a.label == curate.HELD_OUT, "run"].isin(("O3a", "O3b")).all())


if __name__ == "__main__":
    unittest.main()
