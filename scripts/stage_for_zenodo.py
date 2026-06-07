"""Stage a Zenodo deposit bundle from the local data tree.

Produces a single directory with files renamed to the two-tier convention
expected by `scripts/download_data.py` (figures__<rel_path> / full__<rel_path>).
Optionally writes a `manifest.json` with sha256 sums and a `metadata.json`
mirroring `.zenodo.json` for the Zenodo record metadata.

Usage:
    python scripts/stage_for_zenodo.py --tier figures --out ./zenodo_stage_figures
    python scripts/stage_for_zenodo.py --tier full     --out ./zenodo_stage_full \\
        --extra-from <path-to-cluster-rsync-dest>

This script does NOT upload to Zenodo — use the Zenodo web UI or the
`zenodo-client` CLI on the staged directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Tier 0 / 1 — files that live in-repo (Tier 0) plus larger figure-feeding
# artifacts (Tier 1). For now we only have Tier 0 in-repo; Tier 1 expects
# additional files to be present in the data tree (e.g., rsync'd from cluster).
FIGURES_TIER_GLOBS: tuple[str, ...] = (
    # Tier 0 (in-repo)
    "data/topology_g3/best_perm_per_topology.csv",
    "data/topology_g3/best_perm_summary.json",
    "data/topology_g3/topology_graphs.json",
    "data/topology_g3/size_tier_designs/*.json",
    "data/topology_g3/size_tier_designs/*.csv",
    "data/topology_g3/motif_tier_analysis/*.csv",
    "data/topology_g3/motif_tier_analysis/*.json",
    "data/topology_g3/l2_top05/*.csv",
    "data/topology_g3/l2_top05/*.json",
    "data/topology_g3/panel_c_shapley/*.json",
    "data/interp_processed/shapley_taylor_sim_*.json",
    "data/interp_processed/pairwise_*.json",
    # Tier 1 (rsync'd from cluster before staging) — graceful if missing
    "data/topology_g3/design_predictions/**/*.parquet",
    "data/topology_g3/pareto/*.json",
)

# Tier 2 — full back-end artifacts (large; expected to be rsync'd to data/
# from the cluster before staging).
FULL_TIER_GLOBS: tuple[str, ...] = (
    "data/topology_g3/registries/**/*.pkl",
    "data/topology_g3/mlp_checkpoints/**/*.pt",
    "data/topology_g3/population.pkl",
    "data/topology_g3/qc_tiers.pkl",
    "data/exemplars/0x*_design/**/*.h5",
    "data/exemplars/0x*_design/**/*.pt",
    "data/exemplars/0x*_design/**/*.csv",
)


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while buf := f.read(chunk):
            h.update(buf)
    return h.hexdigest()


def _collect(globs: tuple[str, ...]) -> list[Path]:
    out: list[Path] = []
    for g in globs:
        out.extend(sorted(REPO_ROOT.glob(g)))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tier", choices=("figures", "full"), required=True)
    p.add_argument("--out", type=Path, required=True,
                   help="Staging directory to write into.")
    p.add_argument("--metadata", type=Path,
                   default=REPO_ROOT / ".zenodo.json",
                   help="Source Zenodo metadata JSON (default: ./.zenodo.json).")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    if args.tier == "figures":
        candidates = _collect(FIGURES_TIER_GLOBS)
        prefix = "figures__"
    else:
        # Full tier includes everything in figures + the back-end artifacts.
        candidates = _collect(FIGURES_TIER_GLOBS) + _collect(FULL_TIER_GLOBS)
        prefix = "full__"

    if not candidates:
        sys.stderr.write(
            f"No files matched the tier={args.tier} globs. Did you rsync the "
            "Tier 1/2 artifacts from the cluster into data/?\n"
        )
        return 1

    manifest: list[dict] = []
    for src in candidates:
        if not src.is_file():
            continue
        rel = src.relative_to(REPO_ROOT)
        # Flatten path separators into '__' so Zenodo's file UI is browsable.
        key = prefix + str(rel).replace("/", "__")
        dst = args.out / key
        shutil.copy2(src, dst)
        manifest.append({
            "key": key,
            "source_path": str(rel),
            "size_bytes": src.stat().st_size,
            "sha256": _sha256(dst),
        })
        print(f"  staged: {key}  ({src.stat().st_size:>12,} B)")

    # Write the manifest + the Zenodo metadata sidecar.
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True))
    if args.metadata.exists():
        shutil.copy2(args.metadata, args.out / "metadata.json")

    total = sum(m["size_bytes"] for m in manifest)
    print(f"\nStaged {len(manifest)} files, total {total:,} B "
          f"({total/1024/1024:.1f} MiB) to {args.out}")
    print(f"Manifest written to {args.out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
