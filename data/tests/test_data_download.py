"""Offline failure-path tests; no Zenodo access or raw fixtures required."""

import hashlib
from http.client import IncompleteRead
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError

from data import download


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.payload = b"column\nvalue\n"
        self.spec = {
            "name": "H1_O1.csv", "size_bytes": len(self.payload),
            "md5": hashlib.md5(self.payload, usedforsecurity=False).hexdigest(),
            "url": "https://example.invalid/H1_O1.csv",
        }
        self.target = self.output / self.spec["name"]

    def assert_no_partial_files(self):
        self.assertEqual(list(self.output.glob("*.part")), [])

    @patch.object(download, "urlopen")
    def test_verified_file_is_kept_without_network(self, network):
        self.target.write_bytes(self.payload)
        before = self.target.stat().st_mtime_ns
        download.download_file(self.spec, self.output)
        network.assert_not_called()
        self.assertEqual(self.target.stat().st_mtime_ns, before)

    @patch.object(download, "urlopen")
    def test_same_length_corruption_is_repaired(self, network):
        self.target.write_bytes(b"x" * len(self.payload))
        network.return_value = io.BytesIO(self.payload)
        download.download_file(self.spec, self.output)
        self.assertEqual(self.target.read_bytes(), self.payload)
        self.assertTrue(download.verified(self.target, self.spec))
        self.assert_no_partial_files()

    @patch.object(download, "urlopen")
    def test_wrong_checksum_does_not_replace_destination(self, network):
        self.target.write_bytes(b"previous copy")
        network.return_value = io.BytesIO(b"z" * len(self.payload))
        with self.assertRaisesRegex(download.DownloadError, "MD5 mismatch"):
            download.download_file(self.spec, self.output, attempts=1)
        self.assertEqual(self.target.read_bytes(), b"previous copy")
        self.assert_no_partial_files()

    @patch.object(download, "urlopen")
    def test_oversized_response_never_installs(self, network):
        network.return_value = io.BytesIO(self.payload + b"extra")
        with self.assertRaises(download.DownloadError):
            download.download_file(self.spec, self.output, attempts=1)
        self.assertFalse(self.target.exists())
        self.assert_no_partial_files()

    @patch.object(download.time, "sleep")
    @patch.object(download, "urlopen")
    def test_transient_failure_is_retried(self, network, sleep):
        network.side_effect = [URLError("temporary failure"), io.BytesIO(self.payload)]
        download.download_file(self.spec, self.output, attempts=2)
        self.assertEqual(network.call_count, 2)
        self.assertTrue(download.verified(self.target, self.spec))
        self.assert_no_partial_files()

    @patch.object(download.time, "sleep")
    @patch.object(download, "urlopen", side_effect=URLError("unavailable"))
    def test_retries_are_bounded_and_leave_no_files(self, network, sleep):
        with self.assertRaises(download.DownloadError):
            download.download_file(self.spec, self.output, attempts=2)
        self.assertEqual(network.call_count, 2)
        self.assertEqual(list(self.output.iterdir()), [])

    @patch.object(download, "urlopen", side_effect=KeyboardInterrupt)
    def test_keyboard_interrupt_cleans_temporary_file(self, network):
        with self.assertRaises(KeyboardInterrupt):
            download.download_file(self.spec, self.output)
        self.assertEqual(list(self.output.iterdir()), [])

    @patch.object(download, "urlopen")
    def test_incomplete_http_response_is_a_download_failure(self, network):
        class BrokenResponse(io.BytesIO):
            def read(self, size=-1):
                raise IncompleteRead(b"broken", 100)
        network.return_value = BrokenResponse()
        with self.assertRaises(download.DownloadError):
            download.download_file(self.spec, self.output, attempts=1)
        self.assertFalse(self.target.exists())
        self.assert_no_partial_files()

    @patch.object(download, "urlopen")
    def test_verify_only_fails_for_missing_file_without_network(self, network):
        with patch.object(download, "load_manifest", return_value={"files": [self.spec]}):
            result = download.main(["--verify-only", "--output-dir", str(self.output)])
        self.assertEqual(result, 1)
        network.assert_not_called()


if __name__ == "__main__":
    unittest.main()
