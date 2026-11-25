"""Package this E-03 replay losslessly, preserving Windows output fingerprints."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.e03_integrity import sha256_file, verify_complete_artifacts


def main():
    run = ROOT / "results/e03/loco_1125"
    review = ROOT / "results/e03/review_followup_20261125"
    checked = json.loads((review / "worker_checks/aggregate.json").read_text())
    manifest = verify_complete_artifacts(run, repository=ROOT)
    if checked["status"] != "pass" or checked["manifest_sha256"] != sha256_file(run / "manifest.json"):
        raise ValueError("package requires the independently verified complete replay")
    destination = ROOT / "results/e03/loco_1125_archives"
    destination.mkdir(exist_ok=False)
    families = manifest["settings"]["held_outs"]
    groups = {name: [] for name in ["tables", *families]}
    expected = manifest["files"] | {"manifest.json": sha256_file(run / "manifest.json")}
    for relative in sorted(expected):
        parts = Path(relative).parts
        group = "tables"
        if parts[0] in {"predictions", "validation_scores"}:
            group = parts[1].removesuffix(".csv.gz")
        elif parts[0] == "bloch":
            group = next(family for family in families if parts[1].startswith(f"e03-{family}-seed"))
        groups[group].append(relative)
    packages = []
    covered = set()
    for group, relatives in groups.items():
        path = destination / f"loco_1125-{group}.zip"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for relative in relatives:
                info = zipfile.ZipInfo(relative, date_time=(2026, 11, 25, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED if relative.endswith((".gz", ".png")) else zipfile.ZIP_DEFLATED
                with (run / relative).open("rb") as source, archive.open(info, "w") as target:
                    shutil.copyfileobj(source, target, length=1024*1024)
        if path.stat().st_size >= 90*1024*1024:
            raise ValueError(f"archive exceeds publication size bound: {path.name}")
        with zipfile.ZipFile(path) as archive:
            for relative in archive.namelist():
                if relative in covered:
                    raise ValueError("duplicate archive member")
                digest = hashlib.sha256()
                with archive.open(relative) as stream:
                    for block in iter(lambda: stream.read(1024*1024), b""):
                        digest.update(block)
                if digest.hexdigest() != expected[relative]:
                    raise ValueError(f"archive did not preserve original output: {relative}")
                covered.add(relative)
        packages.append(dict(file=path.name, bytes=path.stat().st_size,
                             sha256=sha256_file(path), members=len(relatives)))
        print(path.name, path.stat().st_size, "bytes; all member hashes verified", flush=True)
    if covered != set(expected):
        raise ValueError("incomplete archive coverage")
    receipt = dict(status="pass", experiment="E-03", protocol="P2", tier="L0",
        replay_manifest_sha256=expected["manifest.json"], package_source_sha256=sha256_file(__file__),
        verified_original_files=len(covered), packages=packages,
        extraction="Extract all nine ZIPs into the same new directory, then run experiments.e03_report.")
    (destination / "archives_manifest.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
