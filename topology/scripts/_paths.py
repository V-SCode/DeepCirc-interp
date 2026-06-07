"""Cluster path conventions for the cross-topology pipeline.

This module is the single source of truth for where pipeline outputs live
on `$DEEPCIRC_SCRATCH/`. It encodes two architectural rules:

  1. **Per-topology data is GROUP-AGNOSTIC.** A topology is identified by
     its WL hash (`topology_id`); its trained MLPs, design-space scoring
     output, L1 within-topology rules, and NNGGA input pkl are all
     properties of the topology, not of the substrate (group) it sits in.
     Computed once per topology, reused across G1/G2/G3.

  2. **Per-group analyses are GROUPED.** Population manifest, QC tiers,
     L2 / L3 / L4 outputs (which aggregate across the group's substrate)
     live under `population/<GROUP>/` and are recomputed for each group.

Path layout (post-R5 migration):

    $DEEPCIRC_SCRATCH/
    ├── runs/
    │   ├── stage1/<HEX>/                                    per-target PPO registries
    │   │                                                    (group-agnostic by target)
    │   └── stage2/<topology_id>/                            ← GROUP-AGNOSTIC per-topology MLPs
    │       ├── <circuit_name>_design_0_permutation_*/...
    │       └── _zero_perms_filtered.json (if applicable)
    ├── topology_data/                                       ← GROUP-AGNOSTIC per-topology derived data
    │   ├── nngga/<topology_id>/                             P3 NNGGA input pkls
    │   ├── design_space_predictions/<tid>_<perm>.npz       P7 design-space scoring
    │   └── l1/<topology_id>/perm_<j>.json                  P8 L1 within-topology rules
    └── population/
        ├── G1/                                              ← per-group manifest + cumulative analyses
        │   ├── population.pkl
        │   ├── qc_tiers.pkl
        │   ├── l2/, l3/
        │   ├── best_perm_per_topology.csv
        │   └── design_space_predictions/, l1/, nngga/      ← legacy/deprecated; moved to topology_data/
        └── G2/...

Pre-R5 layout (legacy v1.x / v2.0 G1):

    $DEEPCIRC_SCRATCH/runs/stage2/<GROUP>/<topology_id>/    ← per-group; retrains G1 topos under G2
    $DEEPCIRC_SCRATCH/population/<GROUP>/{nngga, design_space_predictions, l1}/...

The R5 migration script moves data from per-group to group-agnostic paths;
once migrated, all scripts default to the new convention via the helpers
below. Pre-migration, scripts accept legacy --*_root flags as overrides.
"""
from __future__ import annotations

import os
from pathlib import Path


def scratch_root() -> Path:
    """`$DEEPCIRC_SCRATCH` resolved at call time so tests can override it."""
    return Path(os.environ["DEEPCIRC_SCRATCH"])


# ---------------------------------------------------------------------------
# Stage 1 (PPO) — per-target, group-agnostic by target hex
# ---------------------------------------------------------------------------

def stage1_target_dir(target_hex: str) -> Path:
    """`runs/stage1/<HEX>/` — PPO registries per target. Group-agnostic."""
    return scratch_root() / "runs" / "stage1" / target_hex


# ---------------------------------------------------------------------------
# Stage 2 (NNGGA-trained MLPs) — group-agnostic per-topology pool
# ---------------------------------------------------------------------------

def stage2_pool_root() -> Path:
    """`runs/stage2/` — root of the group-agnostic per-topology MLP pool."""
    return scratch_root() / "runs" / "stage2"


def stage2_topology_dir(topology_id: str) -> Path:
    """`runs/stage2/<tid>/` — per-topology trained MLPs.

    Replaces the legacy `runs/stage2/<GROUP>/<tid>/` pattern. A topology's
    MLPs are computed once and reused across any group whose substrate
    contains it.
    """
    return stage2_pool_root() / topology_id


# ---------------------------------------------------------------------------
# Topology-level derived data — group-agnostic per-topology pool
# ---------------------------------------------------------------------------

def topology_data_root() -> Path:
    """`topology_data/` — root of the group-agnostic per-topology derived-data pool."""
    return scratch_root() / "topology_data"


def nngga_pkl_dir(topology_id: str) -> Path:
    """`topology_data/nngga/<tid>/` — P3 NNGGA input pickle dir for one topology.

    Replaces legacy `population/<GROUP>/nngga/<tid>/`.
    """
    return topology_data_root() / "nngga" / topology_id


def design_space_npz(topology_id: str, perm_idx: int) -> Path:
    """`topology_data/design_space_predictions/<tid>_<perm>.npz` — P7 output.

    Replaces legacy `population/<GROUP>/design_space_predictions/<tid>_<perm>.npz`.
    """
    return topology_data_root() / "design_space_predictions" / f"{topology_id}_{perm_idx}.npz"


def design_space_dir() -> Path:
    """`topology_data/design_space_predictions/` — flat dir holding all P7 .npz files."""
    return topology_data_root() / "design_space_predictions"


def l1_topology_dir(topology_id: str) -> Path:
    """`topology_data/l1/<tid>/` — P8 L1 within-topology JSON dir."""
    return topology_data_root() / "l1" / topology_id


def l1_root() -> Path:
    """`topology_data/l1/` — root of the L1 per-topology JSON pool."""
    return topology_data_root() / "l1"


# ---------------------------------------------------------------------------
# Per-group population analyses (manifest + cumulative L2/L3/L4)
# ---------------------------------------------------------------------------

def population_dir(group: str) -> Path:
    """`population/<GROUP>/` — per-group population manifest + cumulative analyses."""
    return scratch_root() / "population" / group


def population_manifest(group: str) -> Path:
    """`population/<GROUP>/population.pkl` — P3 output."""
    return population_dir(group) / "population.pkl"


def qc_tiers_pkl(group: str) -> Path:
    """`population/<GROUP>/qc_tiers.pkl` — P5 output."""
    return population_dir(group) / "qc_tiers.pkl"


def best_perm_csv(group: str) -> Path:
    """`population/<GROUP>/best_perm_per_topology.csv` — P7.5 output."""
    return population_dir(group) / "best_perm_per_topology.csv"


def l2_dir(group: str) -> Path:
    """`population/<GROUP>/l2/` — per-group L2 cumulative outputs (P9 + P12)."""
    return population_dir(group) / "l2"


def l3_dir(group: str) -> Path:
    """`population/<GROUP>/l3/` — per-group L3 cumulative outputs (P10 + P13)."""
    return population_dir(group) / "l3"


# ---------------------------------------------------------------------------
# Migration helpers (used by R5's migrate_to_topology_pool.sh)
# ---------------------------------------------------------------------------

def legacy_stage2_group_dir(group: str) -> Path:
    """`runs/stage2/<GROUP>/` — pre-R5 per-group stage2 dir (legacy)."""
    return scratch_root() / "runs" / "stage2" / group


def legacy_population_subdir(group: str, sub: str) -> Path:
    """`population/<GROUP>/<sub>/` — pre-R5 per-group dir for nngga / design_space_predictions / l1."""
    return population_dir(group) / sub
