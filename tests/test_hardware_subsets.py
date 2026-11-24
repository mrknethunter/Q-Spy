"""D-05 selection, leakage, lifecycle and synthetic local-Git freeze tests."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
import pandas as pd

from data import hardware_subsets as hw
from data import splits
from tests.fixtures.protocol_splits import curated_fixture


def roomy_fixture():
    original = curated_fixture()
    pieces = []
    for replica in range(5):
        part = original.copy()
        part["id"] = part.id + f":replica{replica}"
        part["gps"] += replica * 1000
        pieces.append(part)
    return pd.concat(pieces, ignore_index=True)


class HardwareSubsetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.split_dir = cls.root / "splits"
        cls.frame = roomy_fixture()
        cls.manifest = splits.generate_splits(cls.frame, output_dir=cls.split_dir,
                                              config=splits.SplitConfig(seeds=(0,)))
        cls.payload = hw.generate_hardware_subsets(cls.frame, split_dir=cls.split_dir)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def fresh_directory(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        return Path(temp.name)

    def validate(self, payload, **kwargs):
        hw.validate_hardware_subsets(payload, split_dir=self.split_dir, allow_draft=True, **kwargs)

    def synthetic_freeze(self, *, altered_manifest=False):
        if shutil.which("git") is None:
            self.skipTest("git is required for synthetic freeze tests")
        repo = self.fresh_directory()
        self.git(repo, "init", "-q")
        self.git(repo, "config", "user.name", "Synthetic Unit Test")
        self.git(repo, "config", "user.email", "synthetic-test@example.invalid")
        self.git(repo, "config", "commit.gpgsign", "false")
        (repo / "configs").mkdir()
        (repo / "configs/hardware_frozen.yaml").write_text("purpose: synthetic_unit_test_only\n")
        (repo / "docs").mkdir()
        (repo / "docs/decisions.md").write_text("# Synthetic fixture\nNot a real group approval.\n")
        (repo / "data/splits/v1").mkdir(parents=True)
        content = (self.split_dir / "manifest.json").read_bytes()
        (repo / "data/splits/v1/manifest.json").write_bytes(content + (b"\n" if altered_manifest else b""))
        self.git(repo, "add", ".")
        self.git(repo, "commit", "-q", "-m", "Synthetic test-only L-05 objects")
        self.git(repo, "tag", "-a", "hw-freeze", "-m", "Synthetic unit-test tag")
        return repo

    @staticmethod
    def git(repo, *args):
        return subprocess.run(["git", "-C", str(repo), *args], check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout

    def test_required_sizes_balance_and_held_out(self):
        payload = self.payload
        self.assertEqual(payload["status"], "draft_pending_l05")
        self.assertIsNone(payload["l05_freeze"])
        groups = payload["subsets"]
        for name, size, count in (("main", 120, 15), ("control", 40, 5)):
            self.assertEqual(groups[name]["size"], size)
            self.assertEqual(set(groups[name]["counts_by_class"].values()), {count})
            self.assertEqual(groups[name]["counts_by_detector"], {"H1": size // 2, "L1": size // 2})
            self.assertGreater(groups[name]["counts_by_class"]["Fast_Scattering"], 0)
        for group in groups["concentration"]:
            self.assertEqual(group["size"], 30)
            self.assertEqual(group["counts_by_detector"], {"H1": 15, "L1": 15})
            counts = list(group["counts_by_class"].values())
            self.assertEqual(sorted(counts), [3, 3, 4, 4, 4, 4, 4, 4])
        references = groups["drift_references"]
        self.assertEqual(references["size"], 5)
        self.assertEqual(references["counts_by_class"]["Fast_Scattering"], 1)
        self.assertEqual(sum(x > 0 for x in references["counts_by_class"].values()), 5)

    def test_overlap_policy_and_source_excludes_every_training_validation_id(self):
        groups = self.payload["subsets"]
        main, control = set(groups["main"]["ids"]), set(groups["control"]["ids"])
        concentration = [set(group["ids"]) for group in groups["concentration"]]
        self.assertFalse(main & control)
        self.assertEqual(len(set.union(*concentration)), 90)
        self.assertTrue(set.union(*concentration) <= main)
        self.assertTrue(set(groups["drift_references"]["ids"]) <= main)
        selected = main | control
        lookup = self.frame.set_index("id")
        self.assertEqual(set(lookup.loc[list(selected), "run"]), {"O3b"})
        for entry in self.manifest["splits"]:
            split = splits.load_split(entry["protocol"], entry["seed"], entry["k"], entry["held_out"], split_dir=self.split_dir)
            self.assertFalse(selected & (set(split.train_ids) | set(split.val_ids)))
            if entry["protocol"] in ("P2", "P3"):
                self.assertTrue(selected <= set(split.test_ids))

    def test_same_seed_and_reordered_input_produce_identical_json(self):
        other = hw.generate_hardware_subsets(self.frame.sample(frac=1, random_state=17), split_dir=self.split_dir)
        self.assertEqual(self.payload, other)
        root = self.fresh_directory()
        hw.write_hardware_subsets(self.payload, output=root / "first.json")
        hw.write_hardware_subsets(other, output=root / "second.json")
        self.assertEqual((root / "first.json").read_bytes(), (root / "second.json").read_bytes())

    def test_seed_changes_selection_but_not_cardinality(self):
        other = hw.generate_hardware_subsets(self.frame, split_dir=self.split_dir, seed=2027)
        self.assertNotEqual(self.payload["selection_sha256"], other["selection_sha256"])
        self.assertNotEqual(self.payload["subsets"]["main"]["ids"], other["subsets"]["main"]["ids"])
        self.validate(other)

    def test_draft_can_be_reviewed_but_default_consumer_rejects_it(self):
        output = self.fresh_directory() / "hardware.json"
        hw.write_hardware_subsets(self.payload, output=output)
        self.assertEqual(hw.load_hardware_subsets(output, split_dir=self.split_dir, allow_draft=True), self.payload)
        with self.assertRaisesRegex(hw.HardwareSubsetError, "pending L-05"):
            hw.load_hardware_subsets(output, split_dir=self.split_dir)

    def test_source_feature_changes_and_asset_corruption_fail(self):
        changed = self.frame.copy()
        changed.loc[0, "snr"] += 1
        with self.assertRaisesRegex(hw.HardwareSubsetError, "differs"):
            hw.generate_hardware_subsets(changed, split_dir=self.split_dir)
        directory = self.fresh_directory() / "bad-source"
        shutil.copytree(self.split_dir, directory)
        (directory / "P1.npz").write_bytes(b"broken")
        with self.assertRaisesRegex(hw.HardwareSubsetError, "integrity"):
            hw.generate_hardware_subsets(self.frame, split_dir=directory)

    def test_tampered_ids_or_provenance_fail(self):
        altered = deepcopy(self.payload)
        altered["subsets"]["main"]["ids"][0] = "not-an-event"
        with self.assertRaisesRegex(hw.HardwareSubsetError, "digest"):
            self.validate(altered)
        altered = deepcopy(self.payload)
        altered["source"]["dataset_sha256"] = "0" * 64
        altered["selection_sha256"] = hw._selection_digest(altered)
        with self.assertRaisesRegex(hw.HardwareSubsetError, "source differs"):
            self.validate(altered)

    def test_concentration_overlap_is_rejected_even_with_consistent_counts(self):
        altered = deepcopy(self.payload)
        altered["subsets"]["concentration"][1] = {**deepcopy(altered["subsets"]["concentration"][0]), "name": "concentration_2"}
        altered["selection_sha256"] = hw._selection_digest(altered)
        with self.assertRaisesRegex(hw.HardwareSubsetError, "pairwise-disjoint"):
            self.validate(altered)

    def test_incorrect_counts_and_cardinality_fail(self):
        altered = deepcopy(self.payload)
        altered["subsets"]["control"]["counts_by_detector"]["H1"] += 1
        altered["selection_sha256"] = hw._selection_digest(altered)
        with self.assertRaisesRegex(hw.HardwareSubsetError, "Incorrect counts"):
            self.validate(altered)
        altered = deepcopy(self.payload)
        altered["subsets"]["main"]["ids"].pop()
        altered["selection_sha256"] = hw._selection_digest(altered)
        with self.assertRaisesRegex(hw.HardwareSubsetError, "cardinality"):
            self.validate(altered)

    def test_existing_selection_is_immutable_and_identical_write_is_reused(self):
        output = self.fresh_directory() / "hardware.json"
        hw.write_hardware_subsets(self.payload, output=output)
        before = output.stat().st_mtime_ns
        hw.write_hardware_subsets(self.payload, output=output)
        self.assertEqual(before, output.stat().st_mtime_ns)
        other = hw.generate_hardware_subsets(self.frame, split_dir=self.split_dir, seed=2027)
        with self.assertRaisesRegex(hw.HardwareSubsetError, "selection differs"):
            hw.write_hardware_subsets(other, output=output)

    def test_missing_freeze_does_not_create_or_claim_one(self):
        repo = self.fresh_directory()
        with self.assertRaisesRegex(hw.HardwareSubsetError, "complete L-05"):
            hw.generate_hardware_subsets(self.frame, split_dir=self.split_dir, freeze_repo=repo)
        self.assertFalse((repo / "configs/hardware_frozen.yaml").exists())
        self.assertEqual(self.payload["status"], hw.DRAFT)

    def test_actual_synthetic_tag_binding_preserves_exact_ids(self):
        repo = self.synthetic_freeze()
        bound = hw.generate_hardware_subsets(self.frame, split_dir=self.split_dir, freeze_repo=repo)
        self.assertEqual(bound["status"], "bound_to_hw_freeze")
        self.assertEqual(bound["subsets"], self.payload["subsets"])
        self.assertEqual(bound["selection_sha256"], self.payload["selection_sha256"])
        expected = self.git(repo, "rev-parse", "hw-freeze^{commit}").decode().strip()
        self.assertEqual(bound["l05_freeze"]["commit"], expected)
        output = self.fresh_directory() / "hardware.json"
        hw.write_hardware_subsets(self.payload, output=output)
        hw.write_hardware_subsets(bound, output=output)
        loaded = hw.load_hardware_subsets(output, split_dir=self.split_dir, freeze_repo=repo)
        self.assertEqual(loaded, bound)

    def test_binding_uses_tagged_blobs_not_uncommitted_working_files(self):
        repo = self.synthetic_freeze()
        first = hw.generate_hardware_subsets(self.frame, split_dir=self.split_dir, freeze_repo=repo)
        (repo / "configs/hardware_frozen.yaml").write_text("uncommitted: change\n")
        second = hw.generate_hardware_subsets(self.frame, split_dir=self.split_dir, freeze_repo=repo)
        self.assertEqual(first, second)

    def test_mismatched_tagged_dataset_and_moved_tags_are_rejected(self):
        wrong = self.synthetic_freeze(altered_manifest=True)
        with self.assertRaisesRegex(hw.HardwareSubsetError, "different D-04"):
            hw.generate_hardware_subsets(self.frame, split_dir=self.split_dir, freeze_repo=wrong)
        repo = self.synthetic_freeze()
        bound = hw.generate_hardware_subsets(self.frame, split_dir=self.split_dir, freeze_repo=repo)
        (repo / "configs/hardware_frozen.yaml").write_text("purpose: changed_synthetic_fixture\n")
        self.git(repo, "add", ".")
        self.git(repo, "commit", "-q", "-m", "Synthetic changed tagged config")
        self.git(repo, "tag", "-f", "hw-freeze")
        with self.assertRaisesRegex(hw.HardwareSubsetError, "Stored binding differs"):
            self.validate(bound, freeze_repo=repo)

    def test_invalid_seed_and_forged_lifecycle_status_fail(self):
        for seed in (-1, True, 2**32):
            with self.subTest(seed=seed), self.assertRaises(hw.HardwareSubsetError):
                hw.generate_hardware_subsets(self.frame, split_dir=self.split_dir, seed=seed)
        altered = deepcopy(self.payload)
        altered["status"] = hw.BOUND
        with self.assertRaises(hw.HardwareSubsetError):
            self.validate(altered)

    def test_cli_generates_a_pending_document(self):
        root = self.fresh_directory()
        csv_path = root / "curated.csv"
        output = root / "hardware.json"
        self.frame.to_csv(csv_path, index=False, float_format="%.17g")
        result = hw.main(["--curated-csv", str(csv_path), "--split-dir", str(self.split_dir), "--output", str(output)])
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(output.read_text()), self.payload)


if __name__ == "__main__":
    unittest.main()
