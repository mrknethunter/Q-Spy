"""Offline behavioral checks of all three D-04 protocols and frozen files."""

from dataclasses import replace
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from data import splits
from tests.fixtures.protocol_splits import curated_fixture


class ProtocolSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.directory = cls.root / "frozen"
        cls.frame = curated_fixture()
        cls.lookup = cls.frame.set_index("id")
        cls.manifest = splits.generate_splits(cls.frame, output_dir=cls.directory)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def load(self, protocol, seed=0, k=None, held_out=None):
        return splits.load_split(protocol, seed, k, held_out, split_dir=self.directory)

    def copy_bundle(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name) / "copy"
        shutil.copytree(self.directory, directory)
        return directory

    def test_expected_protocol_grid_has_all_ten_seeds(self):
        entries = self.manifest["splits"]
        self.assertEqual(len(entries), 130)
        for protocol, expected in (("P1", 40), ("P2", 80), ("P3", 10)):
            group = [entry for entry in entries if entry["protocol"] == protocol]
            self.assertEqual(len(group), expected)
            self.assertEqual({entry["seed"] for entry in group}, set(range(10)))

    def test_every_split_has_disjoint_valid_ids_and_accurate_counts(self):
        all_ids = set(self.frame.id)
        for entry in self.manifest["splits"]:
            with self.subTest(protocol=entry["protocol"], seed=entry["seed"], k=entry["k"], held_out=entry["held_out"]):
                result = self.load(entry["protocol"], entry["seed"], entry["k"], entry["held_out"])
                parts = [set(result.train_ids), set(result.val_ids), set(result.test_ids)]
                self.assertFalse(parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])
                self.assertTrue(set.union(*parts) <= all_ids)
                for name, ids in zip(("train", "val", "test"), (result.train_ids, result.val_ids, result.test_ids)):
                    self.assertEqual(len(ids), len(set(ids)))
                    self.assertEqual(ids, tuple(sorted(ids)))
                    self.assertEqual(len(ids), entry["counts"][name]["total"])
                    observed = self.lookup.loc[list(ids)].groupby(["label", "ifo"]).size()
                    for label, detector_counts in entry["counts"][name]["by_class_detector"].items():
                        for ifo, count in detector_counts.items():
                            self.assertEqual(int(observed.get((label, ifo), 0)), count)

    def test_p1_exact_k_balanced_fixed_test_and_validation(self):
        reference = self.load("P1", 0, 5).test_ids
        for seed in range(10):
            val_reference = self.load("P1", seed, 5).val_ids
            previous = set()
            for k in (5, 10, 20, 50):
                result = self.load("P1", seed, k)
                self.assertEqual(result.test_ids, reference)
                self.assertEqual(result.val_ids, val_reference)
                for ids, expected in ((result.train_ids, k), (result.val_ids, 20), (result.test_ids, 100)):
                    selected = self.lookup.loc[list(ids)]
                    self.assertNotIn("Fast_Scattering", set(selected.label))
                    self.assertEqual(selected.groupby("label").size().to_dict(),
                                     {label: expected for label in self.manifest["known_classes"]})
                    for _, group in selected.groupby("label"):
                        self.assertEqual(set(group.ifo), {"H1", "L1"})
                self.assertTrue(previous <= set(result.train_ids))
                previous = set(result.train_ids)
        self.assertNotEqual(self.load("P1", 0, 5).train_ids, self.load("P1", 1, 5).train_ids)

    def test_p2_all_rotations_keep_validation_known_only_and_test_normals(self):
        for held_out in ("Fast_Scattering", *self.manifest["known_classes"]):
            unknown = {"Fast_Scattering", held_out}
            expected_unknown = set(self.frame.loc[self.frame.label.isin(unknown), "id"])
            reference = self.load("P2", 0, held_out=held_out).test_ids
            for seed in range(10):
                result = self.load("P2", seed, held_out=held_out)
                for ids in (result.train_ids, result.val_ids):
                    self.assertFalse(set(self.lookup.loc[list(ids), "label"]) & unknown)
                test = self.lookup.loc[list(result.test_ids)]
                self.assertTrue(expected_unknown <= set(result.test_ids))
                self.assertGreater(int((~test.label.isin(unknown)).sum()), 0)
                self.assertEqual(result.test_ids, reference)
                self.assertEqual(len(result.train_ids) + len(result.val_ids) + len(result.test_ids), len(self.frame))
        default = self.load("P2")
        explicit = self.load("P2", held_out="Fast_Scattering")
        self.assertEqual(default, explicit)
        self.assertEqual((len(default.train_ids), len(default.val_ids), len(default.test_ids)), (1008, 336, 576))
        self.assertNotEqual(self.load("P2", 0).val_ids, self.load("P2", 1).val_ids)

    def test_p3_is_strictly_temporal_with_o3a_only_validation(self):
        reference = None
        for seed in range(10):
            result = self.load("P3", seed)
            for ids in (result.train_ids, result.val_ids):
                group = self.lookup.loc[list(ids)]
                self.assertEqual(set(group.run), {"O3a"})
                self.assertNotIn("Fast_Scattering", set(group.label))
            test = self.lookup.loc[list(result.test_ids)]
            self.assertEqual(set(test.run), {"O3b"})
            self.assertEqual(set(result.test_ids), set(self.frame.loc[self.frame.run == "O3b", "id"]))
            self.assertIn("Fast_Scattering", set(test.label))
            self.assertEqual((len(result.train_ids), len(result.val_ids), len(result.test_ids)), (336, 84, 540))
            if reference is not None:
                self.assertEqual(result.test_ids, reference)
            reference = result.test_ids

    def test_same_data_and_seeds_make_byte_identical_files_even_after_reordering(self):
        with tempfile.TemporaryDirectory() as directory:
            original = self.frame.copy(deep=True)
            other = Path(directory) / "reordered"
            config = splits.SplitConfig(seeds=tuple(reversed(range(10))))
            splits.generate_splits(self.frame.sample(frac=1, random_state=77), output_dir=other, config=config)
            for file in self.directory.iterdir():
                self.assertEqual(file.read_bytes(), (other / file.name).read_bytes(), file.name)
            assert_frame_equal(self.frame, original)

    def test_repeat_generation_reuses_files_without_rewriting(self):
        before = {p.name: p.stat().st_mtime_ns for p in self.directory.iterdir()}
        splits.generate_splits(self.frame, output_dir=self.directory)
        self.assertEqual(before, {p.name: p.stat().st_mtime_ns for p in self.directory.iterdir()})

    def test_existing_version_is_not_replaced_when_config_changes(self):
        target = self.copy_bundle()
        before = {p.name: p.read_bytes() for p in target.iterdir()}
        with self.assertRaisesRegex(splits.SplitError, "version already exists"):
            splits.generate_splits(self.frame, output_dir=target,
                                    config=replace(splits.SplitConfig(), test_pool_seed=2027))
        self.assertEqual(before, {p.name: p.read_bytes() for p in target.iterdir()})

    def test_dataset_verification_binds_features_not_only_ids(self):
        splits.verify_dataset(self.frame.iloc[::-1], split_dir=self.directory)
        for name, value in (("snr", 11.), ("gps", 2.), ("confidence", .95)):
            changed = self.frame.copy()
            changed.loc[0, name] = value
            with self.subTest(name=name), self.assertRaisesRegex(splits.SplitError, "differs"):
                splits.verify_dataset(changed, split_dir=self.directory)

    def test_asset_corruption_is_rejected_before_use(self):
        for name, protocol, k in (("catalog.npz", "P1", 5), ("P1.npz", "P1", 5),
                                  ("P2.npz", "P2", None), ("P3.npz", "P3", None)):
            with self.subTest(name=name):
                directory = self.copy_bundle()
                path = directory / name
                content = bytearray(path.read_bytes())
                content[len(content) // 2] ^= 1
                path.write_bytes(content)
                with self.assertRaisesRegex(splits.SplitError, "integrity"):
                    splits.load_split(protocol, 0, k, split_dir=directory)

    def test_bad_manifest_and_inconsistent_counts_fail(self):
        directory = self.copy_bundle()
        manifest_path = directory / "manifest.json"
        manifest_path.write_text("not json")
        with self.assertRaises(splits.SplitError):
            splits.load_split("P1", 0, 5, split_dir=directory)
        manifest = json.loads((self.directory / "manifest.json").read_text())
        manifest["splits"][0]["counts"]["train"]["total"] += 1
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(splits.SplitError, "counts disagree"):
            splits.load_split("P1", 0, 5, split_dir=directory)

    def test_invalid_selectors_do_not_generate_new_files(self):
        selectors = [("P0", 0, None, None), ("P1", 0, None, None),
                     ("P1", 0, True, None), ("P1", 0, 5.0, None), ("P1", 0, 5, "Blip"),
                     ("P2", 0, 5, None), ("P2", 0, None, "missing"),
                     ("P3", 0, None, "Blip"), ("P1", 10, 5, None), ("P1", True, 5, None)]
        before = {p.name: p.read_bytes() for p in self.directory.iterdir()}
        for args in selectors:
            with self.subTest(args=args), self.assertRaises(splits.SplitError):
                splits.load_split(*args, split_dir=self.directory)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.directory.iterdir()})

    def test_missing_version_does_not_generate_a_replacement(self):
        with self.assertRaises(splits.SplitError):
            splits.load_split("P1", 0, 5, split_dir=self.root / "does-not-exist")
        self.assertFalse((self.root / "does-not-exist").exists())

    def test_npz_files_have_fixed_timestamps_and_nonobject_arrays(self):
        for name in splits.FILES:
            with zipfile.ZipFile(self.directory / name) as archive:
                self.assertEqual(archive.namelist(), sorted(archive.namelist()))
                for entry in archive.infolist():
                    self.assertEqual(entry.date_time, (1980, 1, 1, 0, 0, 0))
            with np.load(self.directory / name, allow_pickle=False) as archive:
                for key in archive.files:
                    self.assertFalse(archive[key].dtype.hasobject)

    def test_invalid_datasets_are_rejected(self):
        cases = [self.frame.drop(columns="snr"), self.frame.iloc[:0],
                 pd.concat([self.frame, self.frame.iloc[:1]], ignore_index=True),
                 self.frame.assign(ifo="V1"), self.frame.assign(run="O4"),
                 self.frame.assign(snr=np.inf), self.frame.assign(confidence=1.1),
                 self.frame.assign(is_held_out=False)]
        for frame in cases:
            with self.subTest(shape=frame.shape), self.assertRaises(splits.SplitError):
                splits.generate_splits(frame, output_dir=self.root / "invalid")
        bad_fast = self.frame.copy()
        bad_fast.loc[bad_fast.label == "Fast_Scattering", "run"] = "O2"
        with self.assertRaisesRegex(splits.SplitError, "confined"):
            splits.generate_splits(bad_fast, output_dir=self.root / "invalid")

    def test_insufficient_samples_fail_instead_of_sampling_with_replacement(self):
        small = self.frame.loc[~((self.frame.label == "Blip") & self.frame.id.str.endswith(tuple(f":{i:03d}" for i in range(40, 120))))]
        with self.assertRaisesRegex(splits.SplitError, "Insufficient detector support"):
            splits.generate_splits(small, output_dir=self.root / "small")
        no_o3a = self.frame.loc[~((self.frame.label == "Blip") & (self.frame.ifo == "H1") & (self.frame.run == "O3a"))]
        with self.assertRaisesRegex(splits.SplitError, "P3 needs O3a"):
            splits.generate_splits(no_o3a, output_dir=self.root / "no-o3a")

    def test_config_validation(self):
        invalid = [dict(seeds=()), dict(seeds=(0, 0)), dict(seeds=(True,)), dict(seeds=(-1,)),
                   dict(test_per_class=1), dict(val_per_class=1), dict(test_pool_seed=-1),
                   dict(p2_val_fraction=np.nan), dict(p2_val_fraction=.8, p2_test_fraction=.3),
                   dict(p3_val_fraction=1)]
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(splits.SplitError):
                splits.SplitConfig(**config)

    def test_p1_sampling_keeps_detector_proportions(self):
        unequal = self.frame.loc[~((self.frame.ifo == "L1") & self.frame.id.str.endswith(tuple(f":{i:03d}" for i in range(80, 120))))]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "unequal"
            splits.generate_splits(unequal, output_dir=directory, config=splits.SplitConfig(seeds=(0,)))
            split = splits.load_split("P1", 0, 5, split_dir=directory)
            counts = self.lookup.loc[list(split.test_ids)].groupby(["label", "ifo"]).size()
            for label in self.manifest["known_classes"]:
                self.assertEqual((counts[(label, "H1")], counts[(label, "L1")]), (60, 40))

    def test_cli_reads_round_trip_curated_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "curated.csv"
            self.frame.to_csv(path, index=False, float_format="%.17g")
            result = splits.main(["--curated-csv", str(path), "--output-dir", str(root / "v1")])
            self.assertEqual(result, 0)
            self.assertEqual((root / "v1" / "manifest.json").read_bytes(), (self.directory / "manifest.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
