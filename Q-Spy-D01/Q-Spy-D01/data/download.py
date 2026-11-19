"""Fetch the pinned Gravity Spy release; Python 3.10+, standard library only.

Run from the repository root: python data/download.py
"""

from __future__ import annotations

import argparse
import hashlib
from http.client import HTTPException
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

DATA_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = DATA_DIR / "manifest.json"
DEFAULT_OUTPUT = DATA_DIR / "raw"
CHUNK_SIZE = 1024 * 1024


class DownloadError(Exception):
    """A file could not be fetched or did not match the pinned release."""


def load_manifest() -> dict:
    with MANIFEST_PATH.open(encoding="utf-8") as source:
        return json.load(source)


def md5sum(path: Path) -> str:
    # MD5 matches the publisher's checksum; it is not an authenticity guarantee.
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified(path: Path, spec: dict) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == spec["size_bytes"]
        and md5sum(path) == spec["md5"]
    )


def download_file(spec: dict, output: Path, *, attempts: int = 3,
                  timeout: float = 60.0) -> Path:
    """Keep valid files; install replacements only after size and MD5 checks.

    A failed/interrupted transfer removes its own temporary file and leaves
    any previous destination untouched. Partial files are never accepted.
    """
    output.mkdir(parents=True, exist_ok=True)
    target = output / spec["name"]
    if verified(target, spec):
        print(f"OK   {target.name} (already verified)", flush=True)
        return target

    if target.exists():
        print(f"REPAIR {target.name} (size/checksum mismatch)", flush=True)

    for attempt in range(1, attempts + 1):
        temporary = None
        try:
            print(f"GET  {target.name} ({spec['size_bytes']:,} bytes; "
                  f"attempt {attempt}/{attempts})", flush=True)
            request = Request(spec["url"], headers={"User-Agent": "Q-Spy-D01/1.0"})
            digest = hashlib.md5(usedforsecurity=False)
            size = 0
            with tempfile.NamedTemporaryFile(
                dir=output, prefix=f".{target.name}.", suffix=".part", delete=False
            ) as stream:
                temporary = Path(stream.name)
                with urlopen(request, timeout=timeout) as response:
                    for chunk in iter(lambda: response.read(CHUNK_SIZE), b""):
                        size += len(chunk)
                        if size > spec["size_bytes"]:
                            raise DownloadError(f"{target.name}: response exceeds pinned size")
                        digest.update(chunk)
                        stream.write(chunk)
            if size != spec["size_bytes"] or digest.hexdigest() != spec["md5"]:
                raise DownloadError(f"{target.name}: size or MD5 mismatch")
            os.replace(temporary, target)
            print(f"OK   {target.name} (size and MD5 verified)", flush=True)
            return target
        except (OSError, URLError, HTTPException, DownloadError) as error:
            if attempt == attempts:
                raise DownloadError(f"{target.name}: failed after {attempts} attempt(s): {error}") from error
            print(f"RETRY {target.name}: {error}", file=sys.stderr, flush=True)
            time.sleep(min(2 ** (attempt - 1), 8))
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    raise DownloadError(f"{target.name}: no download attempts configured")


def main(argv: list[str] | None = None) -> int:
    manifest = load_manifest()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--files", nargs="+", choices=[s["name"] for s in manifest["files"]],
                        help="Download a subset; omit for all eight files.")
    parser.add_argument("--verify-only", action="store_true",
                        help="Verify local files without any network requests.")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="Socket timeout in seconds (default: 60).")
    args = parser.parse_args(argv)
    if args.attempts < 1 or args.timeout <= 0:
        parser.error("--attempts and --timeout must be positive")
    specs = [s for s in manifest["files"] if args.files is None or s["name"] in args.files]
    try:
        if args.verify_only:
            invalid = []
            for spec in specs:
                valid = verified(args.output_dir / spec["name"], spec)
                print(f"{'OK  ' if valid else 'FAIL'} {spec['name']}", flush=True)
                if not valid:
                    invalid.append(spec["name"])
            if invalid:
                raise DownloadError("Missing or corrupt files: " + ", ".join(invalid))
        else:
            for spec in specs:
                download_file(spec, args.output_dir, attempts=args.attempts, timeout=args.timeout)
    except (DownloadError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Download interrupted. Re-run the same command to continue.", file=sys.stderr)
        return 130
    print(f"Verified {len(specs)}/{len(specs)} requested files in {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
