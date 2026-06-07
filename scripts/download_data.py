"""Fetch DeepCirc-interp data from the Zenodo deposit.

Two tiers:
  --tier figures   ~10 MB of small CSV/JSON intermediates needed to rebuild
                   the published S10–S15 PDFs without re-running the back-end.
  --tier full      adds trained MLP checkpoints, design-space predictions,
                   per-exemplar HDF5 samples, registries — tens of GB,
                   needed for full re-runs from population-scoring onward.

The Zenodo record is identified by `--doi` (default below). Override at the
command line:

    python scripts/download_data.py --tier figures --doi 10.5281/zenodo.XXXXXX

Files are written under `--root` (default ./data), preserving the layout the
analysis + figure scripts expect.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

DEFAULT_DOI = "10.XXXX/XXXXX"  # placeholder — replace at submission
ZENODO_API = "https://zenodo.org/api/records/{record_id}"


def _resolve_record_id(doi: str) -> str:
    """Extract the bare record id from a Zenodo DOI like '10.5281/zenodo.123456'."""
    if "zenodo." in doi:
        return doi.split("zenodo.")[-1].strip()
    raise ValueError(f"Could not parse Zenodo record id from DOI: {doi!r}")


def _fetch_manifest(record_id: str) -> dict:
    url = ZENODO_API.format(record_id=record_id)
    with urllib.request.urlopen(url) as r:
        return json.load(r)


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while buf := f.read(chunk):
            h.update(buf)
    return h.hexdigest()


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"  → {dst.relative_to(Path.cwd())}", flush=True)
    urllib.request.urlretrieve(url, dst)


def _filter_by_tier(files: list[dict], tier: str) -> list[dict]:
    """Subset the Zenodo file list by tier tag in filename.

    Convention: files in the 'figures' tier are prefixed `figures__`, files in
    the 'full' tier are prefixed `full__`. The Zenodo deposit upload script
    follows this naming convention.
    """
    prefixes = {"figures": ("figures__",), "full": ("figures__", "full__")}[tier]
    return [f for f in files if f["key"].startswith(prefixes)]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tier", choices=("figures", "full"), default="figures")
    p.add_argument("--doi", default=DEFAULT_DOI)
    p.add_argument("--root", type=Path, default=Path("./data"))
    p.add_argument("--dry-run", action="store_true",
                   help="List what would be downloaded without fetching.")
    args = p.parse_args()

    if args.doi == DEFAULT_DOI:
        sys.stderr.write(
            "ERROR: Zenodo DOI is still the placeholder value. Set --doi or\n"
            "       edit DEFAULT_DOI in this script after the deposit is minted.\n"
        )
        return 2

    print(f"Resolving Zenodo record for DOI {args.doi}...", flush=True)
    record_id = _resolve_record_id(args.doi)
    rec = _fetch_manifest(record_id)
    files = _filter_by_tier(rec.get("files", []), args.tier)
    if not files:
        sys.stderr.write(f"No files matched tier={args.tier}.\n")
        return 1

    print(f"Will fetch {len(files)} files for tier={args.tier}:", flush=True)
    for f in files:
        # Strip the tier prefix so the on-disk path matches the script-expected layout.
        rel = f["key"].split("__", 1)[1]
        dst = args.root / rel
        if args.dry_run:
            print(f"  [dry] {dst}")
        else:
            _download(f["links"]["self"], dst)
            got = _sha256(dst)
            want = f["checksum"].split(":")[-1] if "checksum" in f else None
            if want and got != want:
                sys.stderr.write(
                    f"SHA256 mismatch for {dst.name}: got {got}, want {want}\n"
                )
                return 1
    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
