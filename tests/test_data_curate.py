"""D-02: filtering, coincidence leakage, deterministic sampling and I/O."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from data import curate
from tests.fixtures.curation import extra_row, raw_fixture, write_release


class CurationTests(unittest.TestCase):
    def with_rows(self, *rows, config=None):
        return curate.curate_frame(pd.concat([raw_fixture(), pd.DataFrame(rows)], ignore_index=True), config)

    def test_threshold_is_inclusive_and_rejects_invalid_probabilities(self):
        additions = [extra_row(f"edge-{i}", peak_time=1_500_000_000 + i,
                               ml_confidence=value) for i, value in enumerate(
                                   (0.899999, 0.9, 1., 1.01, -1, np.nan, np.inf, "bad"))]
        result = self.with_rows(*additions)
        self.assertEqual(set(result.loc[result.id.str.contains("edge"), "id"]),
                         {"H1:edge-1", "H1:edge-2"})
        # Raising the threshold must not silently return a partial class set.
        with self.assertRaisesRegex(curate.CurationError, "both detectors"):
            self.with_rows(*additions, config=curate.CurateConfig(confidence_threshold=1.))

    def test_empty_or_incomplete_class_support_fails(self):
        raw = raw_fixture()
        with self.assertRaisesRegex(curate.CurationError, "No usable rows"):
            curate.curate_frame(raw.assign(ml_confidence=0.1))
        with self.assertRaisesRegex(curate.CurationError, "both detectors"):
            curate.curate_frame(raw.loc[~((raw.ml_label == "Blip") & (raw.ifo == "L1"))])
        with self.assertRaisesRegex(curate.CurationError, "both detectors"):
            curate.curate_frame(raw.loc[raw.ml_label != curate.HELD_OUT])

    def test_configurable_threshold(self):
        additions = [extra_row("low", ml_confidence=.8)]
        default = self.with_rows(*additions)
        relaxed = self.with_rows(*additions, config=curate.CurateConfig(confidence_threshold=.8))
        self.assertNotIn("H1:low", set(default.id))
        self.assertIn("H1:low", set(relaxed.id))

    def test_held_out_is_nonempty_and_restricted_to_o3(self):
        result = self.with_rows(extra_row("bad-fast", ml_label=curate.HELD_OUT, run="O2"))
        held = result.loc[result.is_held_out]
        self.assertGreater(len(held), 0)
        self.assertEqual(set(held.label), {curate.HELD_OUT})
        self.assertTrue(held.run.isin(("O3a", "O3b")).all())
        self.assertNotIn("H1:bad-fast", set(result.id))
        self.assertEqual(result.attrs["curation"]["removed"]["held_out_outside_o3"], 1)

    def test_quality_rejections_and_counts(self):
        overrides = [
            {"duration": 0}, {"peak_frequency": -1}, {"amplitude": np.inf},
            {"bandwidth": "bad"}, {"central_freq": np.nan}, {"q_value": -np.inf},
            {"snr": 0}, {"peak_time_ns": 1_000_000_000}, {"peak_time_ns": .5},
            {"peak_time": -1}, {"peak_time": 1.5}, {"peak_time": 10_000_000_000},
            {"gravityspy_id": None}, {"ml_label": None},
        ]
        result = self.with_rows(*[extra_row(f"invalid-{i}", **kw) for i, kw in enumerate(overrides)])
        self.assertEqual(len(result), len(raw_fixture()))
        report = result.attrs["curation"]
        self.assertEqual(report["removed"]["invalid_id_label_time_or_feature"], len(overrides))
        self.assertEqual(report["input_rows"] - sum(report["removed"].values()), len(result))

    def test_extreme_q_choice_is_explicit_and_reversible(self):
        row = extra_row("extreme", q_value=1e20)
        self.assertNotIn("H1:extreme", set(self.with_rows(row).id))
        result = self.with_rows(row, config=curate.CurateConfig(drop_extreme_q=False))
        self.assertEqual(result.loc[result.id == "H1:extreme", "q_value"].item(), 1e20)

    def test_id_duplicates_choose_confidence_then_snr(self):
        row = raw_fixture().iloc[0].to_dict()
        a = {**row, "ml_confidence": .999, "snr": 12.}
        b = {**row, "ml_confidence": .999, "snr": 13.}
        result = self.with_rows(a, b)
        kept = result.loc[result.id == "H1:" + row["gravityspy_id"]]
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept.snr.item(), 13.)
        self.assertEqual(result.attrs["curation"]["removed"]["repeated_id"], 2)

    def test_nanosecond_window_boundary_is_exact(self):
        for gap, expected in ((100_000_000, 1), (100_000_001, 2)):
            with self.subTest(gap=gap):
                result = self.with_rows(extra_row("time-a", peak_time_ns=0),
                                        extra_row("time-b", peak_time_ns=gap))
                self.assertEqual(int(result.id.str.contains("time-").sum()), expected)

    def test_equal_quality_conflicting_labels_do_not_depend_on_row_order(self):
        a = extra_row("conflict", ml_label="Tomte")
        b = extra_row("conflict", ml_label="Blip")
        raw = pd.concat([raw_fixture(), pd.DataFrame([a, b])], ignore_index=True)
        first = curate.curate_frame(raw)
        reverse = curate.curate_frame(raw.iloc[::-1])
        assert_frame_equal(first, reverse)
        self.assertEqual(first.loc[first.id == "H1:conflict", "label"].item(), "Blip")

    def test_exact_only_mode_preserves_nearby_distinct_times(self):
        result = self.with_rows(extra_row("time-a", peak_time_ns=0),
                                extra_row("time-b", peak_time_ns=1),
                                config=curate.CurateConfig(coincidence_window_s=0))
        self.assertEqual(int(result.id.str.contains("time-").sum()), 2)

    def test_transitive_cluster_and_deterministic_tie(self):
        result = self.with_rows(extra_row("cluster-z", peak_time_ns=0),
                                extra_row("cluster-a", peak_time_ns=90_000_000),
                                extra_row("cluster-c", peak_time_ns=180_000_000))
        self.assertEqual(list(result.loc[result.id.str.contains("cluster"), "id"]), ["H1:cluster-a"])

    def test_deduplication_ignores_label_and_precedes_selection(self):
        result = self.with_rows(extra_row("selected", peak_time_ns=0),
                                extra_row("excluded", peak_time_ns=0,
                                          ml_label="No_Glitch", ml_confidence=.999))
        self.assertNotIn("H1:selected", set(result.id))
        self.assertNotIn("H1:excluded", set(result.id))
        self.assertEqual(result.attrs["curation"]["removed"]["unselected_class"], 1)

    def test_detectors_remain_distinct_and_run_boundary_does_not_hide_duplicates(self):
        result = self.with_rows(extra_row("coincidence", ifo="H1", run="O1"),
                                extra_row("coincidence", ifo="L1", run="O1"),
                                extra_row("other-run", ifo="H1", run="O2", ml_confidence=.999))
        self.assertIn("L1:coincidence", set(result.id))
        self.assertIn("H1:other-run", set(result.id))
        self.assertNotIn("H1:coincidence", set(result.id))

    def test_sampling_is_stratified_without_replacement_and_order_invariant(self):
        config = curate.CurateConfig(max_per_class=8, seed=123)
        raw = raw_fixture()
        original = raw.copy(deep=True)
        a = curate.curate_frame(raw, config)
        b = curate.curate_frame(raw.sample(frac=1, random_state=1), config)
        assert_frame_equal(a, b)
        assert_frame_equal(raw, original)
        self.assertTrue(a.id.is_unique)
        for label in (*curate.KNOWN_CLASSES, curate.HELD_OUT):
            self.assertEqual(a.attrs["curation"]["counts_by_detector"][label], {"H1": 6, "L1": 2})
        c = curate.curate_frame(raw, replace(config, seed=321))
        self.assertNotEqual(set(a.id), set(c.id))

    def test_small_caps_and_uncapped_small_classes_retain_both_detectors(self):
        raw = raw_fixture()
        result = curate.curate_frame(raw, curate.CurateConfig(max_per_class=2))
        for counts in result.attrs["curation"]["counts_by_detector"].values():
            self.assertEqual(counts, {"H1": 1, "L1": 1})
        assert_frame_equal(curate.curate_frame(raw),
                           curate.curate_frame(raw, curate.CurateConfig(max_per_class=1000)))

    def test_invalid_provenance_and_schema_fail(self):
        raw = raw_fixture()
        for column, value in (("ifo", "V1"), ("run", "O4"), ("ifo", None), ("run", None)):
            with self.subTest(column=column, value=value):
                bad = raw.copy()
                bad.loc[0, column] = value
                with self.assertRaisesRegex(curate.CurationError, "provenance"):
                    curate.curate_frame(bad)
        with self.assertRaisesRegex(curate.CurationError, "Missing columns"):
            curate.curate_frame(raw.drop(columns="q_value"))

    def test_config_validation(self):
        bad = [dict(confidence_threshold=np.nan), dict(confidence_threshold=1.1),
               dict(confidence_threshold=True), dict(coincidence_window_s=-1),
               dict(coincidence_window_s=np.inf), dict(max_per_class=1), dict(max_per_class=True),
               dict(seed=-1), dict(seed=2**32), dict(seed=True), dict(drop_extreme_q="yes"),
               dict(known_classes=curate.KNOWN_CLASSES[:5]),
               dict(known_classes=(*curate.KNOWN_CLASSES, curate.HELD_OUT)),
               dict(known_classes=("Blip",) * 6)]
        for kwargs in bad:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                curate.CurateConfig(**kwargs)


class CurationIOTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = write_release(self.root / "raw")

    def test_loader_verifies_all_sources_before_reading_any(self):
        (self.root / "raw" / "L1_O3b.csv").write_text("corrupt")
        with patch.object(curate, "load_manifest", return_value=self.manifest), \
                patch.object(curate.pd, "read_csv") as read:
            with self.assertRaisesRegex(curate.CurationError, "missing/corrupt"):
                curate.load_curated(raw_dir=self.root / "raw")
            read.assert_not_called()

    def test_manifest_must_be_complete(self):
        self.manifest["files"].pop()
        with patch.object(curate, "load_manifest", return_value=self.manifest):
            with self.assertRaisesRegex(curate.CurationError, "eight"):
                curate.load_curated(raw_dir=self.root / "raw")

    def test_wrong_file_detector_and_missing_columns_fail_even_with_valid_checksum(self):
        for content, message in (("ifo\nH1\n", "Usecols"),
                                 (None, "detector column")):
            with self.subTest(message=message):
                manifest = write_release(self.root / "raw")
                path = self.root / "raw" / "H1_O1.csv"
                if content is None:
                    table = pd.read_csv(path)
                    table["ifo"] = "L1"
                    table.to_csv(path, index=False)
                else:
                    path.write_text(content)
                import hashlib
                spec = manifest["files"][0]
                spec.update(size_bytes=path.stat().st_size,
                            md5=hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest())
                with patch.object(curate, "load_manifest", return_value=manifest):
                    with self.assertRaisesRegex(curate.CurationError, message):
                        curate.load_curated(raw_dir=self.root / "raw")

    def test_reports_are_consistent_and_data_card_preserves_d01(self):
        with patch.object(curate, "load_manifest", return_value=self.manifest):
            frame = curate.load_curated(raw_dir=self.root / "raw")
        card = self.root / "docs" / "data_card.md"
        card.parent.mkdir()
        prefix = "# Original D-01 card\n\nKeep this original text.\n"
        card.write_text(prefix)
        kwargs = dict(output_dir=self.root / "curated", report_path=self.root / "docs" / "report.json", card_path=card)
        curate.write_outputs(frame, **kwargs)
        first = card.read_bytes()
        curate.write_outputs(frame, **kwargs)
        self.assertEqual(card.read_bytes(), first)
        self.assertTrue(card.read_text().startswith(prefix))
        self.assertEqual(card.read_text().count(curate.CARD_BEGIN), 1)
        report = json.loads(kwargs["report_path"].read_text())
        self.assertEqual(sum(sum(v.values()) for v in report["counts_by_detector"].values()), len(frame))
        self.assertEqual(report["input_rows"] - sum(report["removed"].values()), len(frame))
        saved = pd.read_csv(self.root / "curated" / "curated.csv")
        self.assertEqual(len(saved), len(frame))
        self.assertEqual(list(saved.columns), list(curate.OUTPUT_COLUMNS))

    def test_malformed_card_markers_do_not_write_outputs(self):
        frame = curate.curate_frame(raw_fixture())
        card = self.root / "card.md"
        card.write_text(curate.CARD_BEGIN)
        with self.assertRaisesRegex(curate.CurationError, "markers"):
            curate.write_outputs(frame, output_dir=self.root / "out",
                                 report_path=self.root / "report.json", card_path=card)
        self.assertFalse((self.root / "out").exists())

    def test_cli_generates_dataset_and_reports(self):
        with patch.object(curate, "load_manifest", return_value=self.manifest):
            result = curate.main([
                "--raw-dir", str(self.root / "raw"), "--max-per-class", "8",
                "--output-dir", str(self.root / "out"),
                "--report", str(self.root / "report.json"), "--data-card", str(self.root / "card.md"),
            ])
        self.assertEqual(result, 0)
        report = json.loads((self.root / "report.json").read_text())
        self.assertEqual(report["output_rows"], 64)
        self.assertTrue((self.root / "out" / "curated.csv").is_file())


if __name__ == "__main__":
    unittest.main()
