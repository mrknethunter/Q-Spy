"""Check statistics on small hand-calculated fixtures, including bad values."""

import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from data import profile


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "H1_O1.csv"

    def write_rows(self, confidence, ifo="H1"):
        columns = [*profile.FEATURES, "ifo", "ml_label", "ml_confidence", "channel"]
        with self.path.open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=columns)
            writer.writeheader()
            for index, value in enumerate(confidence):
                row = {name: index + 1 for name in profile.FEATURES}
                row.update(ifo=ifo, ml_label="Blip" if index % 2 == 0 else "Tomte",
                           ml_confidence=value, channel="GDS-CALIB_STRAIN")
                writer.writerow(row)

    def test_counts_ranges_and_interpolated_quantiles(self):
        self.write_rows([0, 0.5, 0.9, 1])
        result = profile.scan_csv(self.path, "H1")
        self.assertEqual(result["rows"], 4)
        self.assertEqual(result["classes"], {"Blip": 2, "Tomte": 2})
        self.assertEqual(result["features"]["duration"]["minimum"], 1)
        self.assertEqual(result["features"]["duration"]["maximum"], 4)
        self.assertAlmostEqual(result["confidence"]["p50"], 0.7)
        self.assertAlmostEqual(result["confidence"]["p10"], 0.15)
        self.assertAlmostEqual(result["confidence"]["mean"], 0.6)
        self.assertEqual(sum(result["confidence"]["histogram"].values()), 4)

    def test_histogram_edges_and_invalid_values_are_accounted_for(self):
        self.write_rows([0, 0.5, 0.8, 0.9, 0.95, 0.99, 1, "", "nan", "inf", "bad", -0.1, 1.1])
        c = profile.scan_csv(self.path, "H1")["confidence"]
        self.assertEqual(list(c["histogram"].values()), [1, 1, 1, 1, 1, 2])
        self.assertEqual(c["out_of_range"], 2)
        self.assertEqual(c["missing"], 1)
        self.assertEqual(c["invalid"], 1)
        self.assertEqual(c["nonfinite"], 2)
        self.assertEqual(c["finite"], 9)

    def test_numeric_quality_counts_do_not_hide_bad_inputs(self):
        stats = profile.NumericStats()
        for raw in ("", "bad", "NaN", "inf", "-inf", "0", "-2", "3"):
            stats.add(raw)
        self.assertEqual((stats.finite, stats.missing, stats.invalid, stats.nonfinite), (3, 1, 1, 3))
        self.assertEqual(stats.nonpositive, 2)
        self.assertEqual((stats.minimum, stats.maximum), (-2, 3))

    def test_zero_confidence_and_extreme_q_are_retained(self):
        self.write_rows([0])
        text = self.path.read_text(encoding="utf-8")
        lines = text.splitlines()
        columns = lines[0].split(",")
        row = lines[1].split(",")
        row[columns.index("q_value")] = "1e20"
        self.path.write_text(lines[0] + "\n" + ",".join(row) + "\n", encoding="utf-8")
        result = profile.scan_csv(self.path, "H1")
        self.assertEqual(result["rows"], 1)
        self.assertEqual(result["features"]["q_value"]["maximum"], 1e20)
        self.assertEqual(result["quality_flags"]["q_value_equals_1e20"], 1)
        self.assertEqual(result["quality_flags"]["confidence_zero"], 1)

    def test_unexpected_detector_fails(self):
        self.write_rows([0.9], ifo="L1")
        with self.assertRaisesRegex(ValueError, "unexpected ifo"):
            profile.scan_csv(self.path, "H1")

    def test_missing_schema_fails(self):
        self.path.write_text("ifo,ml_label\nH1,Blip\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing columns"):
            profile.scan_csv(self.path, "H1")

    def test_partial_dataset_is_refused_before_profiling(self):
        with patch.object(profile, "scan_csv") as scan:
            with self.assertRaisesRegex(ValueError, "missing/corrupt"):
                profile.build_summary(Path(self.temporary.name))
            scan.assert_not_called()


if __name__ == "__main__":
    unittest.main()
