from pathlib import Path

import pytest

SPLITS = Path(__file__).resolve().parent.parent / "data" / "splits"


@pytest.mark.parametrize("path", sorted(SPLITS.rglob("*.json")), ids=lambda p: p.name)
def test_hash_pinned_split_files_keep_lf_line_endings(path):
    # D-04/D-05 verify these files by SHA-256 of their bytes: a Windows CRLF checkout breaks the check
    assert b"\r" not in path.read_bytes(), f"{path} has CRLF line endings: see .gitattributes"
