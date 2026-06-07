"""29 — Single-body Shapley for the 30 designs in figS04 Panel C.

For each of the 30 designs (15 max-circ per-target + 15 knee per-target,
both at paper MLP-gate thresholds), compute per-slot Shapley contributions
to circuit_score and growth_score MLP predictions.

  Φ_k(d) = MLP(d) − E_{p valid at slot k}[MLP(d with p at slot k)]

Where "valid" enforces the no-repeated-protein constraint of the design
manifold. For a k-reg design we enumerate ALL 21−k valid replacements
per slot exactly (no sampling needed — it's tiny).

Output JSON: data/G3/panel_c_shapley/shapley_per_design.json.
Pulled by figures/setB_design_arc/figS05/scripts/
build_panel_d_stacked.py to render figS05 Panel D.

Run on cluster (CPU is plenty — MLP is 10×100 dense):

    # TODO: substitute your cluster login + scratch paths
    ssh <USER>@<your-cluster-login>
    cd <REPO_ROOT> && git pull
    salloc -p <partition> -c 4 --mem=16G -t 0:30:00
    module purge && module load miniforge/24.3.0-0
    conda activate $HOME/envs/deepcirc
    python topology/scripts/29_panel_c_shapley.py
    git add data/topology_g3/panel_c_shapley/
    git commit -m "P29: single-body Shapley for figS05 Panel D"
    git push origin main
"""
from __future__ import annotations

import ast
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "topology" / "figures"))
import _loaders as L  # noqa: E402

_SCRATCH_ENV = os.environ.get("DEEPCIRC_SCRATCH")
if not _SCRATCH_ENV:
    raise SystemExit(
        "DEEPCIRC_SCRATCH is not set. Export it before running, e.g.:\n"
        "    export DEEPCIRC_SCRATCH=$HOME/deepcirc_scratch   # TODO: set to your cluster scratch path\n"
    )
SCRATCH = Path(_SCRATCH_ENV)
RUNS_DIR = SCRATCH / "runs" / "stage2"
POP_PATH = SCRATCH / "population" / "G3" / "population.pkl"

OUT_DIR = REPO_ROOT / "data" / "topology_g3" / "panel_c_shapley"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Paper MLP-gate thresholds (= figS04 Panel C buildable definition).
CIRC_SCORE_FLOOR = 2.0
CIRC_LOG_FLOOR   = float(np.log(CIRC_SCORE_FLOOR))
TOX_FLOOR        = 0.5

LIBRARY_SIZE = 20   # paper Methods: 20-part TetR library


# ---------- design selection (mirrors figures/.../build_panel_c.py) ----

def _parse_source_list(src) -> list[str]:
    s = str(src).strip()
    if s.startswith("["):
        try:
            return list(ast.literal_eval(s))
        except Exception:
            return [s]
    return [s]


def _top15_maxcirc() -> pd.DataFrame:
    """Top 15 max-circuit per-target buildable designs."""
    fronts = L.load_pareto_fronts("G3").copy()
    fronts["circuit_score"] = np.exp(fronts["circuit_log"])
    mask = ((fronts["circuit_score"] >= CIRC_SCORE_FLOOR)
            & (fronts["toxicity"] >= TOX_FLOOR))
    buildable = fronts[mask].copy()
    buildable["sources_list"] = buildable["source"].apply(_parse_source_list)
    rows = []
    for _, r in buildable.iterrows():
        for tgt in r["sources_list"]:
            rows.append({**r.to_dict(), "target": tgt})
    long = pd.DataFrame(rows)
    best = (long.sort_values("circuit_log", ascending=False)
                .groupby("target", as_index=False)
                .first())
    best = best.sort_values("circuit_log", ascending=False).head(15).copy()
    best["group"] = "max-circ"
    return best


def _top15_knee() -> pd.DataFrame:
    """Top 15 knee per-target by combined_score at paper-MLP-gate floors."""
    k = L.load_pareto_knees("G3").copy()
    mask = ((k["knee_A_circuit_log"] >= CIRC_LOG_FLOOR)
            & (k["knee_A_toxicity"]   >= TOX_FLOOR))
    k = k[mask].copy()
    circ_range = float(k["knee_A_circuit_log"].max() - CIRC_LOG_FLOOR)
    tox_range  = 1.0 - TOX_FLOOR
    k["combined_score"] = (
        (k["knee_A_circuit_log"] - CIRC_LOG_FLOOR) / circ_range
        + (k["knee_A_toxicity"]  - TOX_FLOOR)      / tox_range
    )
    k["sources_list"] = k["source"].apply(_parse_source_list)
    rows = []
    for _, r in k.iterrows():
        for tgt in r["sources_list"]:
            rows.append({**r.to_dict(), "target": tgt})
    long = pd.DataFrame(rows)
    best = (long.sort_values("combined_score", ascending=False)
                .groupby("target", as_index=False)
                .first())
    best = (best.sort_values("combined_score", ascending=False)
                .head(15).copy())
    # Normalise columns to the max-circ schema so downstream code is uniform.
    best["circuit_log"]      = best["knee_A_circuit_log"]
    best["toxicity"]         = best["knee_A_toxicity"]
    best["circuit_score"]    = np.exp(best["circuit_log"])
    best["part_names"]       = best["knee_A_part_names"]
    best["gate_assignments"] = best["knee_A_gate_assignments"]
    best["perm_idx"]         = best["knee_A_perm"]
    best["group"]            = "knee"
    return best


# ---------- MLP loader -------------------------------------------------------

def build_mlp(k_regs: int) -> nn.Sequential:
    """Mirror the upstream RegressionNN architecture.

    From state_dict inspection: 10 dense layers, 9 hidden width=100 with
    ReLU between, final width=1 linear output. Input dim = k * 20."""
    layers: list[nn.Module] = []
    in_dim = k_regs * LIBRARY_SIZE
    for _ in range(9):
        layers.append(nn.Linear(in_dim, 100))
        layers.append(nn.ReLU())
        in_dim = 100
    layers.append(nn.Linear(in_dim, 1))
    return nn.Sequential(*layers)


def load_mlp(ckpt_path: Path, k_regs: int) -> nn.Sequential:
    """Load an upstream MLP state_dict into a nn.Sequential. Strips the
    "model." key prefix so the loaded weights line up with our layer
    indices."""
    state = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    new_state = {kk.replace("model.", ""): vv for kk, vv in state.items()}
    mlp = build_mlp(k_regs)
    mlp.load_state_dict(new_state)
    mlp.eval()
    return mlp


# ---------- Shapley ---------------------------------------------------------

def gates_to_onehot(gates: list[int], k_regs: int) -> torch.Tensor:
    oh = torch.zeros(k_regs, LIBRARY_SIZE)
    for slot, g in enumerate(gates):
        oh[slot, int(g)] = 1.0
    return oh.flatten()


def slot_shapleys(mlp: nn.Sequential, gates: list[int], k_regs: int
                   ) -> tuple[float, list[float]]:
    """Per-slot single-body Shapley.

    For each slot k:
      base   = MLP(d)
      mean_p = mean over all valid replacements p of MLP(d with p at slot k)
      Φ_k    = base − mean_p

    Valid replacements = parts NOT already used at other slots (no-
    repeated-protein constraint, k_regs ≤ LIBRARY_SIZE).
    """
    with torch.no_grad():
        x  = gates_to_onehot(gates, k_regs).unsqueeze(0)
        base = float(mlp(x).item())
        phis: list[float] = []
        for slot in range(k_regs):
            other = {int(g) for i, g in enumerate(gates) if i != slot}
            valid = [p for p in range(LIBRARY_SIZE) if p not in other]
            batch = torch.stack([
                gates_to_onehot(
                    [(p if i == slot else int(g)) for i, g in enumerate(gates)],
                    k_regs,
                )
                for p in valid
            ])
            mean_alt = float(mlp(batch).mean().item())
            phis.append(base - mean_alt)
        return base, phis


# ---------- main ------------------------------------------------------------

def main() -> None:
    print(f"Loading population manifest from {POP_PATH} ...")
    with open(POP_PATH, "rb") as f:
        pop = pickle.load(f)
    cn_by_topo = {t["topology_id"]: t.get("nngga_circuit_name")
                   for t in pop["topologies"]}
    print(f"  {sum(1 for v in cn_by_topo.values() if v)} topologies have nngga_circuit_name")

    designs = pd.concat([_top15_maxcirc(), _top15_knee()], ignore_index=True)
    print(f"Total designs to score: {len(designs)} "
          f"({(designs['group']=='max-circ').sum()} max-circ + "
          f"{(designs['group']=='knee').sum()} knee)")

    out: dict = {
        "thresholds": {
            "circuit_score_floor": CIRC_SCORE_FLOOR,
            "growth_floor":        TOX_FLOOR,
        },
        "library_size": LIBRARY_SIZE,
        "designs": [],
    }

    skipped = 0
    for i, r in designs.iterrows():
        topo = r["topology_id"]
        perm = int(r["perm_idx"])
        cn   = cn_by_topo.get(topo)
        if cn is None:
            print(f"  [{i+1}/{len(designs)}] SKIP {topo}: no nngga_circuit_name")
            skipped += 1
            continue

        run_dir = RUNS_DIR / topo / f"{cn}_design_0_permutation_{perm}"
        c_ckpt = run_dir / "circuit_score_model.pt"
        g_ckpt = run_dir / "toxicity_score_model.pt"
        if not c_ckpt.exists() or not g_ckpt.exists():
            print(f"  [{i+1}/{len(designs)}] SKIP {topo} perm {perm}: "
                  f"missing {c_ckpt.name} or {g_ckpt.name}")
            skipped += 1
            continue

        gates = (ast.literal_eval(r["gate_assignments"])
                 if isinstance(r["gate_assignments"], str)
                 else list(r["gate_assignments"]))
        k_regs = len(gates)

        circ_mlp = load_mlp(c_ckpt, k_regs)
        grow_mlp = load_mlp(g_ckpt, k_regs)

        c_base, phi_c = slot_shapleys(circ_mlp, gates, k_regs)
        g_base, phi_g = slot_shapleys(grow_mlp, gates, k_regs)

        out["designs"].append({
            "group":             r["group"],
            "target":            r["target"],
            "topology_id":       topo,
            "perm_idx":          perm,
            "gate_count":        int(r["gate_count"]),
            "circuit_log":       float(r["circuit_log"]),
            "toxicity":          float(r["toxicity"]),
            "part_names": (ast.literal_eval(r["part_names"])
                           if isinstance(r["part_names"], str)
                           else list(r["part_names"])),
            "gate_assignments":  [int(g) for g in gates],
            "mlp_circuit_score": c_base,
            "mlp_growth_score":  g_base,
            "phi_circuit":       phi_c,
            "phi_growth":        phi_g,
        })

        if (i + 1) % 5 == 0 or (i + 1) == len(designs):
            print(f"  [{i+1}/{len(designs)}] done: {topo} k={k_regs} "
                  f"c_base={c_base:.3g} g_base={g_base:.3f}")

    out_path = OUT_DIR / "shapley_per_design.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {len(out['designs'])} designs (skipped {skipped}) "
          f"to {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
