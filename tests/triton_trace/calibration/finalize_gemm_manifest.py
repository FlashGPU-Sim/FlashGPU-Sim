"""Update provenance hashes after the launch harness has been built."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = REPO_ROOT / "reference" / "SM89_RTX4060" / "gemm"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--dtype", choices=("fp16",), required=True)
    parser.add_argument("--split", choices=("calibration", "holdout", "diagnostic", "full_set"), required=True)
    args = parser.parse_args()
    case_dir = ROOT / args.split / args.dtype / args.case
    manifest_path = case_dir / "experiment-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_roots = (case_dir / "binaries", case_dir / "data", case_dir / "launchers")
    artifacts = {}
    for artifact_root in artifact_roots:
        if artifact_root.exists():
            for path in sorted(artifact_root.rglob("*")):
                if path.is_file() and path.suffix not in {".o", ".a"}:
                    artifacts[str(path.relative_to(case_dir))] = sha256(path)
    manifest["artifacts"] = artifacts
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"MANIFEST_FINALIZED={manifest_path}")
    print(f"HASHED_ARTIFACTS={len(artifacts)}")


if __name__ == "__main__":
    main()
