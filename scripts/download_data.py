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

Two deposit layouts are supported:
  (1) Individual tier-tagged files (`figures__*`, `full__*`): matches the
      script's native convention; files are downloaded one by one with
      SHA256 verification.
  (2) A single source-archive zip (e.g. `DeepCirc-interp-vX.Y.Z.zip` from
      GitHub's auto-attached release archive): falls back when no tier-
      tagged files are present. The zip is downloaded, extracted, and the
      bundled `data/` subtree is copied into `--root`. This matches the
      v1.0 upload layout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_DOI = "10.5281/zenodo.20576709"
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


def _fallback_source_archive(all_files: list[dict], root: Path,
                              dry_run: bool) -> int:
    """Deposit holds no tier-tagged files — try the GitHub-attached
    source zip (`*-vX.Y.Z.zip`), extract it, and copy its bundled `data/`
    subtree into `root`."""
    zips = [f for f in all_files if f["key"].lower().endswith(".zip")]
    if not zips:
        sys.stderr.write(
            "No tier-tagged files AND no .zip in the deposit. "
            "Check the DOI points at the right record.\n"
        )
        return 1
    archive = zips[0]
    print(f"Falling back to source archive: {archive['key']}", flush=True)
    if dry_run:
        print(f"  [dry] download + extract to {root}")
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / archive["key"]
        _download(archive["links"]["self"], zip_path)
        print(f"  extracting ...", flush=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        # Zenodo source archives extract into a single top-level folder
        # (e.g. `V-SCode-DeepCirc-interp-abc123/`). Find the `data/`
        # subtree inside it.
        tmp_root = Path(tmp)
        data_dirs = list(tmp_root.rglob("data"))
        data_dirs = [d for d in data_dirs if d.is_dir()
                      and (d / "topology_g3").exists()]
        if not data_dirs:
            sys.stderr.write(
                "Extracted archive but no `data/topology_g3/` found inside.\n"
            )
            return 1
        src = data_dirs[0]
        root.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            dst = root / item.name
            if dst.exists():
                print(f"  skip (already present): {dst}")
                continue
            if item.is_dir():
                shutil.copytree(item, dst)
            else:
                shutil.copy2(item, dst)
            print(f"  → {dst}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tier", choices=("figures", "full"), default="figures")
    p.add_argument("--doi", default=DEFAULT_DOI)
    p.add_argument("--root", type=Path, default=Path("./data"))
    p.add_argument("--dry-run", action="store_true",
                   help="List what would be downloaded without fetching.")
    args = p.parse_args()

    print(f"Resolving Zenodo record for DOI {args.doi}...", flush=True)
    record_id = _resolve_record_id(args.doi)
    rec = _fetch_manifest(record_id)
    all_files = rec.get("files", [])
    files = _filter_by_tier(all_files, args.tier)
    if not files:
        print(f"No `{args.tier}__*` tier-tagged files in the deposit.",
              flush=True)
        return _fallback_source_archive(all_files, args.root, args.dry_run)

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
