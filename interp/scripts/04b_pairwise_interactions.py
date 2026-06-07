#!/usr/bin/env python3
"""Module B extension — full-K Shapley main + Shapley-Taylor pairwise interactions.

Upgrades over `04_yellow_dot_local_analysis.py`:
  1. Valid-completion marginal is drawn from the **full valid-K set** (9.35M
     for 0x6D; full training-data set for 0x2B/0x17 — 87,552 / 873,936 rows).
  2. Shapley-sample count bumped from 80 → 400 for tighter CIs on main effects.
  3. Adds Shapley-Taylor 2nd-order interaction index Φ_{ij} per slot pair (i,j),
     capturing super- / sub-additive compatibility between slots.

Shapley-Taylor 2nd-order (Sundararajan et al. 2020) per pair (i, j):
  Φ_{ij} = sum_{S ⊆ N\\{i,j}} w(|S|) [v(S∪{i,j}) − v(S∪{i}) − v(S∪{j}) + v(S)]
with w(s) = s! (n-s-2)! / (n-1)! — so the coalition weighting emphasises mid-size S.

We sample coalitions uniformly from N\\{i,j} and compute the 4 value-function
terms per sample; v(S) is the MLP prediction averaged over n_marginal draws
of valid completions (same on-manifold convention as Module B).

Outputs:
  processed/pairwise_{name}.json     — main effects + pairwise matrix (circuit + growth)
  figures/pairwise_{name}.png        — 4-panel: main + pairwise heatmap per metric
  reports/pairwise_interactions.md   — narrative
"""
from __future__ import annotations

import json
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loaders as L
from plotting_utils import apply_exploratory_style, save_figure_with_data

apply_exploratory_style()

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "processed"
FIGS = ROOT / "figures"
REPORTS = ROOT / "reports"
FIGS.mkdir(parents=True, exist_ok=True)


def load_completion_pool(name: str) -> np.ndarray:
    """Return a (M, l) int64 array of valid permutations for completion marginal."""
    if name == "0x6D":
        # Use the full valid-K HDF5 (9.35M unique after dedup)
        ex = L.EXEMPLARS[name]
        h5p = ex.dir / "sampled_valid_permutations_7_gates_10000000_samples.h5"
        df = pd.read_hdf(h5p, key="df").drop_duplicates().reset_index(drop=True)
        return df.values.astype(np.int64)
    # For 0x2B / 0x17, use the full training-data set
    td = L.load_training_data(name)
    return np.stack([np.asarray(p, dtype=np.int64) for p in td["permutation"]])


def v_of_S_batch(model, perm, pool, S_maskS, n_marginal: int, rng):
    """Compute v(S) = E_completion [ f(x_S = yd[S], x_{-S} ~ pool) ]
    for a BATCH of masks S (each a length-l boolean array).

    Uses a shared set of marginal draws across all masks for variance reduction.
    Returns an array v of length len(S_maskS).
    """
    idx = rng.integers(0, len(pool), size=n_marginal)
    drawn = pool[idx]  # (n_marginal, l)
    vs = np.empty(len(S_maskS), dtype=np.float64)
    for i, mask in enumerate(S_maskS):
        out = drawn.copy()
        out[:, mask] = perm[mask]
        preds = L.predict(model, out, batch_size=max(n_marginal, 1024))
        vs[i] = float(preds.mean())
    return vs


def main_shapley(perm, model, pool, n_samples=400, n_marginal=32, rng=None) -> np.ndarray:
    """Monte-Carlo Shapley main effect per slot (1D array length l)."""
    if rng is None:
        rng = np.random.default_rng(0)
    l = len(perm)
    perm = np.asarray(perm, dtype=np.int64)
    shap = np.zeros(l, dtype=np.float64)
    for _ in range(n_samples):
        order = rng.permutation(l)
        mask = np.zeros(l, dtype=bool)
        idx = rng.integers(0, len(pool), size=n_marginal)
        drawn = pool[idx]
        v_prev = L.predict(model, drawn, batch_size=1024).mean()
        for p in order:
            mask[p] = True
            out = drawn.copy()
            out[:, mask] = perm[mask]
            v_new = float(L.predict(model, out, batch_size=1024).mean())
            shap[p] += v_new - v_prev
            v_prev = v_new
    return shap / n_samples


def shapley_taylor_pair(perm, model, pool, i, j,
                        n_samples=150, n_marginal=32, rng=None) -> float:
    """Shapley-Taylor 2nd-order pair interaction Φ_{ij}.

    Φ_{ij} = E_{S ⊂ N\\{i,j}} [v(S∪{i,j}) − v(S∪{i}) − v(S∪{j}) + v(S)]
    with S drawn uniformly over subsets of N\\{i,j} (this corresponds to the
    symmetric Monte-Carlo weighting; exact Shapley-Taylor weights emphasise
    balanced coalition sizes via w(s)).
    """
    if rng is None:
        rng = np.random.default_rng(1000 + i * 100 + j)
    l = len(perm)
    perm = np.asarray(perm, dtype=np.int64)
    others = [k for k in range(l) if k not in (i, j)]
    acc = 0.0
    for _ in range(n_samples):
        # Random subset size s ∈ {0..l-2}, then choose s elements of `others`
        s_size = rng.integers(0, len(others) + 1)
        sel = rng.choice(others, size=s_size, replace=False) if s_size > 0 else np.array([], dtype=int)
        m_base = np.zeros(l, dtype=bool)
        m_base[sel] = True
        mask_S = m_base.copy()
        mask_Si = m_base.copy(); mask_Si[i] = True
        mask_Sj = m_base.copy(); mask_Sj[j] = True
        mask_Sij = m_base.copy(); mask_Sij[i] = True; mask_Sij[j] = True
        # Shared marginal draws
        idx = rng.integers(0, len(pool), size=n_marginal)
        drawn = pool[idx]
        def v(mask):
            out = drawn.copy()
            out[:, mask] = perm[mask]
            return float(L.predict(model, out, batch_size=1024).mean())
        acc += v(mask_Sij) - v(mask_Si) - v(mask_Sj) + v(mask_S)
    return acc / n_samples


def analyze_one(name: str) -> dict:
    print(f"\n[{name}] loading completion pool ...")
    pool = load_completion_pool(name)
    print(f"  pool size: {len(pool):,}")
    yd = L.get_yellow_dot_info(name)
    perm = np.asarray(yd["perm"], dtype=np.int64)
    l = len(perm)

    cm = L.load_circuit_mlp(name)
    gm = L.load_growth_mlp(name)

    rng = np.random.default_rng(42)
    print(f"  main Shapley (circuit) ...")
    shap_c_main = main_shapley(perm, cm, pool, n_samples=400, n_marginal=32, rng=rng)
    rng2 = np.random.default_rng(43)
    print(f"  main Shapley (growth) ...")
    shap_g_main = main_shapley(perm, gm, pool, n_samples=400, n_marginal=32, rng=rng2)

    print(f"  Shapley-Taylor pairwise — {l*(l-1)//2} pairs ...")
    pair_c = np.zeros((l, l), dtype=np.float64)
    pair_g = np.zeros((l, l), dtype=np.float64)
    for i, j in combinations(range(l), 2):
        pair_c[i, j] = pair_c[j, i] = shapley_taylor_pair(perm, cm, pool, i, j,
                                                          n_samples=150, n_marginal=32)
        pair_g[i, j] = pair_g[j, i] = shapley_taylor_pair(perm, gm, pool, i, j,
                                                          n_samples=150, n_marginal=32)

    slots = L.regulator_slot_metadata(name)
    out = {
        "name": name,
        "perm": perm.tolist(),
        "pool_size": int(len(pool)),
        "pool_source": ("full_valid_K_hdf5" if name == "0x6D" else "full_training_data"),
        "slot_labels": [f"{r.repressor}-{r.rbs}" for _, r in slots.iterrows()],
        "main_shapley_circuit": shap_c_main.tolist(),
        "main_shapley_growth": shap_g_main.tolist(),
        "pairwise_circuit": pair_c.tolist(),
        "pairwise_growth": pair_g.tolist(),
    }
    with open(PROC / f"pairwise_{name}.json", "w") as f:
        json.dump(out, f, indent=2)

    # --- Figure: 2x2 (main eff + pair heatmap per metric) ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    labels = [f"s{r.slot_idx}\n{r.repressor}\n{r.rbs}" for _, r in slots.iterrows()]

    ax = axes[0, 0]
    ax.bar(range(l), shap_c_main, color=["#3a7" if v >= 0 else "#c33" for v in shap_c_main])
    ax.set_xticks(range(l)); ax.set_xticklabels(labels, fontsize=8)
    ax.axhline(0, color="k", lw=0.4)
    ax.set_title(f"{name} main Shapley (circuit)")
    for i, v in enumerate(shap_c_main):
        ax.text(i, v, f"{v:+.2f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8)

    ax = axes[0, 1]
    pc_display = pair_c.copy()
    np.fill_diagonal(pc_display, 0.0)
    vmax = max(abs(pc_display).max(), 1e-9)
    im = ax.imshow(pc_display, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(l)); ax.set_xticklabels(labels, fontsize=7, rotation=0)
    ax.set_yticks(range(l)); ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(f"{name} Shapley-Taylor pair Φ_ij (circuit)")
    for i in range(l):
        for j in range(l):
            if i != j:
                ax.text(j, i, f"{pc_display[i,j]:+.2f}", ha="center", va="center",
                        fontsize=6, color="k")
    plt.colorbar(im, ax=ax)

    ax = axes[1, 0]
    ax.bar(range(l), shap_g_main, color=["#3a7" if v >= 0 else "#c33" for v in shap_g_main])
    ax.set_xticks(range(l)); ax.set_xticklabels(labels, fontsize=8)
    ax.axhline(0, color="k", lw=0.4)
    ax.set_title(f"{name} main Shapley (growth)")
    for i, v in enumerate(shap_g_main):
        ax.text(i, v, f"{v:+.3f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8)

    ax = axes[1, 1]
    pg_display = pair_g.copy()
    np.fill_diagonal(pg_display, 0.0)
    vmax = max(abs(pg_display).max(), 1e-9)
    im = ax.imshow(pg_display, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(l)); ax.set_xticklabels(labels, fontsize=7, rotation=0)
    ax.set_yticks(range(l)); ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(f"{name} Shapley-Taylor pair Φ_ij (growth)")
    for i in range(l):
        for j in range(l):
            if i != j:
                ax.text(j, i, f"{pg_display[i,j]:+.3f}", ha="center", va="center",
                        fontsize=6, color="k")
    plt.colorbar(im, ax=ax)

    fig.suptitle(f"Yellow-dot pairwise interactions — {name}\n"
                 f"perm={perm.tolist()}, pool={out['pool_source']} (N={len(pool):,})")
    fig.tight_layout()
    save_figure_with_data(
        fig, FIGS / f"pairwise_{name}",
        source_files=[
            (f"processed/pairwise_{name}.json",
             "main + pairwise Shapley-Taylor coefficients for circuit and growth"),
        ],
        how_to_replot="main bars + pair heatmap from the JSON's "
                      "main_shapley_{circuit,growth} and pairwise_{circuit,growth}",
        script_name="04b_pairwise_interactions.py", dpi=140,
    )
    plt.close(fig)

    return out


def main():
    results = []
    for name in L.EXEMPLARS:
        results.append(analyze_one(name))

    with open(REPORTS / "pairwise_interactions.md", "w") as f:
        f.write("# Module B extension — full-K Shapley + Shapley-Taylor pairwise\n\n")
        f.write("Main Shapley values (n_samples=400) plus Shapley-Taylor 2nd-order\n")
        f.write("pairwise interaction Φ_ij per slot pair (n_samples=150 per pair).\n\n")
        f.write("Completion marginal draws:\n")
        f.write("- 0x2B, 0x17: full training-data set (87,552 / 873,936 valid permutations)\n")
        f.write("- 0x6D: full valid-K HDF5 (9,345,706 unique permutations)\n\n")
        for o in results:
            f.write(f"## {o['name']}\n\n")
            slot_lab = o["slot_labels"]
            f.write("**Main effects (Shapley main, circuit):**\n\n")
            f.write("| slot | part | Φ_i (circuit) | Φ_i (growth) |\n|---|---|---|---|\n")
            for i, lab in enumerate(slot_lab):
                f.write(f"| s{i} | {lab} | {o['main_shapley_circuit'][i]:+.3f} | "
                        f"{o['main_shapley_growth'][i]:+.4f} |\n")
            pc = np.array(o["pairwise_circuit"])
            pg = np.array(o["pairwise_growth"])
            f.write("\n**Top pairwise interactions (circuit score):**\n\n")
            pairs = []
            l = pc.shape[0]
            for i, j in combinations(range(l), 2):
                pairs.append((i, j, pc[i, j]))
            pairs.sort(key=lambda t: -abs(t[2]))
            f.write("| pair | parts | Φ_ij (circuit) |\n|---|---|---|\n")
            for i, j, v in pairs[:6]:
                f.write(f"| (s{i}, s{j}) | {slot_lab[i]} × {slot_lab[j]} | {v:+.3f} |\n")
            f.write("\n**Top pairwise interactions (growth):**\n\n")
            pairs = []
            for i, j in combinations(range(l), 2):
                pairs.append((i, j, pg[i, j]))
            pairs.sort(key=lambda t: -abs(t[2]))
            f.write("| pair | parts | Φ_ij (growth) |\n|---|---|---|\n")
            for i, j, v in pairs[:6]:
                f.write(f"| (s{i}, s{j}) | {slot_lab[i]} × {slot_lab[j]} | {v:+.4f} |\n")
            f.write(f"\nFigure: `figures/pairwise_{o['name']}.png`\n\n")
    print(f"wrote {REPORTS/'pairwise_interactions.md'}")


if __name__ == "__main__":
    main()
