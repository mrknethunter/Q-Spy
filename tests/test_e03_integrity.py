"""E-03 artifact safety tests; stdlib unittest runs without Qiskit/sklearn."""
from __future__ import annotations

import ast
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from experiments.e03_integrity import (
    REQUIRED_OUTPUTS, SOURCE_FILES, artifact_path, atomic_json, case_index,
    conservative_threshold, sha256_file, verify_complete_artifacts,
    verify_hashes, verify_restart_identity,
)


class E03IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="e03-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def make_run(self):
        outputs = {*REQUIRED_OUTPUTS, "roc_fast_scattering.png",
                   "predictions/Fast_Scattering.csv.gz", "validation_scores/Fast_Scattering.csv.gz"}
        for relative in outputs:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic integrity fixture; no scientific results")
        case = {"held_out": "Fast_Scattering", "seed": 0}
        (self.root / "audit.json").write_text(json.dumps([case]), encoding="utf-8")
        with gzip.open(self.root / "selected_ids.json.gz", "wt", encoding="utf-8") as stream:
            json.dump([case], stream)
        manifest = dict(experiment="E-03", protocol="P2", tier="L0", status="complete",
                        completed_cases=1, settings={"held_outs": ["Fast_Scattering"], "seeds": [0]},
                        files={name: sha256_file(self.root / name) for name in outputs})
        atomic_json(self.root / "manifest.json", manifest)
        return manifest

    def test_all_e03_sources_are_parseable(self):
        repository = Path(__file__).resolve().parents[1]
        paths = [*(repository / "experiments").glob("e03*.py"),
                 *(repository / "tests").glob("test_e03*.py")]
        for path in sorted(paths):
            with self.subTest(path=path.name):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_complete_fixture_passes_byte_checks(self):
        self.make_run()
        self.assertEqual(verify_complete_artifacts(self.root)["completed_cases"], 1)

    def test_failed_or_partial_runs_are_rejected(self):
        manifest = self.make_run()
        for state in ("running", "failed", "unknown"):
            with self.subTest(state=state):
                atomic_json(self.root / "manifest.json", manifest | {"status": state})
                with self.assertRaisesRegex(ValueError, "only complete"):
                    verify_complete_artifacts(self.root)

    def test_missing_required_output_cannot_be_hidden_by_omitting_its_hash(self):
        manifest = self.make_run()
        del manifest["files"]["selected_ids.json.gz"]
        atomic_json(self.root / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "omits required"):
            verify_complete_artifacts(self.root)

    def test_missing_predictions_are_rejected(self):
        self.make_run()
        (self.root / "predictions/Fast_Scattering.csv.gz").unlink()
        with self.assertRaisesRegex(ValueError, "missing file"):
            verify_complete_artifacts(self.root)

    def test_changed_output_bytes_are_rejected(self):
        self.make_run()
        (self.root / "e03_metrics.csv").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            verify_complete_artifacts(self.root)

    def test_duplicate_audit_records_are_rejected_even_if_hashes_match(self):
        manifest = self.make_run()
        path = self.root / "audit.json"
        cases = json.loads(path.read_text())
        path.write_text(json.dumps(cases * 2), encoding="utf-8")
        manifest["files"]["audit.json"] = sha256_file(path)
        atomic_json(self.root / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "duplicated case"):
            verify_complete_artifacts(self.root)

    def test_duplicate_selections_are_rejected_even_if_hashes_match(self):
        manifest = self.make_run()
        path = self.root / "selected_ids.json.gz"
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            json.dump([{"held_out": "Fast_Scattering", "seed": 0}] * 2, stream)
        manifest["files"]["selected_ids.json.gz"] = sha256_file(path)
        atomic_json(self.root / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "duplicated case"):
            verify_complete_artifacts(self.root)

    def test_missing_seed_or_wrong_completed_count_is_rejected(self):
        manifest = self.make_run()
        atomic_json(self.root / "manifest.json", manifest | {"completed_cases": 2})
        with self.assertRaisesRegex(ValueError, "completed-case"):
            verify_complete_artifacts(self.root)

    def test_case_index_rejects_missing_and_duplicated_seed(self):
        expected = {("Blip", 0), ("Blip", 1)}
        for records in ([{"held_out": "Blip", "seed": 0}],
                        [{"held_out": "Blip", "seed": 0}] * 2):
            with self.assertRaises(ValueError):
                case_index(records, expected)

    def test_path_traversal_and_windows_paths_are_rejected(self):
        for value in ("../other", "/absolute", "C:/other", "predictions\\other", "./other", "", "."):
            with self.subTest(path=value), self.assertRaises(ValueError):
                artifact_path(self.root, value)

    def test_hash_verifier_rejects_empty_mapping(self):
        with self.assertRaises(ValueError):
            verify_hashes(self.root, {})

    def test_restart_requires_identical_data_splits_sources_settings_and_seeds(self):
        previous = dict(dataset_sha256="data", split_manifest_sha256="splits",
                        source_sha256={"e03.py": "code"}, versions={"numpy": "pinned"},
                        parameters={"seeds": [0, 1], "budget": 8, "cv_folds": 3})
        verify_restart_identity(previous, json.loads(json.dumps(previous)))
        for key in previous:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                verify_restart_identity(previous, previous | {key: "changed"})

    def test_threshold_is_conservative_for_every_small_sample_size(self):
        for size in range(1, 501):
            scores = [float(i) for i in range(size)]
            threshold = conservative_threshold(scores)
            self.assertLessEqual(sum(score > threshold for score in scores) / size, .01)

    def test_threshold_ties_constant_scores_and_numerical_extremes(self):
        for scores in ([2.] * 300, [0.] * 99 + [1.], [-1e300, 0., 1e300], [1e-300]):
            threshold = conservative_threshold(scores)
            self.assertLessEqual(sum(score > threshold for score in scores) / len(scores), .01)
        self.assertEqual(conservative_threshold([2.] * 300), 2.)

    def test_threshold_rejects_nonfinite_scores_and_empty_validation(self):
        for scores in ([], [float("nan")], [float("inf")], [-float("inf")]):
            with self.assertRaises(ValueError):
                conservative_threshold(scores)
        for fpr in (0, 1, float("nan")):
            with self.assertRaises(ValueError):
                conservative_threshold([1., 2.], fpr)

    def test_atomic_manifest_replaces_whole_json_without_leftover_tempfiles(self):
        path = self.root / "manifest.json"
        atomic_json(path, {"status": "running"})
        atomic_json(path, {"status": "failed", "error": "synthetic worker failure"})
        self.assertEqual(json.loads(path.read_text())["status"], "failed")
        self.assertEqual([p.name for p in self.root.iterdir()], ["manifest.json"])

    def test_source_fingerprints_cover_adapter_runner_and_configuration(self):
        self.assertIn("experiments/e03_integrity.py", SOURCE_FILES)
        self.assertIn("experiments/e03_preprocessing.py", SOURCE_FILES)
        self.assertIn("experiments/e03_report.py", SOURCE_FILES)
        self.assertIn("experiments/rq4/e03_gravityspy.yaml", SOURCE_FILES)
        self.assertEqual(len(SOURCE_FILES), len(set(SOURCE_FILES)))


if __name__ == "__main__":
    unittest.main()
