"""Local L-02 signature checks; the actual shared L-02 suite was not supplied."""

from dataclasses import FrozenInstanceError
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from data import splits
from tests.fixtures.protocol_splits import curated_fixture


class SplitContractTests(unittest.TestCase):
    def test_four_argument_loader_returns_three_immutable_id_partitions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "v1"
            splits.generate_splits(curated_fixture(), output_dir=root, config=splits.SplitConfig(seeds=(0,)))
            with patch.object(splits, "DEFAULT_SPLIT_DIR", root):
                results = [splits.load_split("P1", 0, 5, None),
                           splits.load_split("P2", 0, None, "Fast_Scattering"),
                           splits.load_split("P3", 0, None, None)]
            for result in results:
                self.assertIsInstance(result, splits.Split)
                for name in ("train_ids", "val_ids", "test_ids"):
                    ids = getattr(result, name)
                    self.assertIsInstance(ids, tuple)
                    self.assertGreater(len(ids), 0)
                    self.assertTrue(all(isinstance(item, str) for item in ids))
                with self.assertRaises(FrozenInstanceError):
                    result.train_ids = ()


if __name__ == "__main__":
    unittest.main()
