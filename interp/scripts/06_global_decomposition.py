#!/usr/bin/env python3
"""Module D — Global rule extraction via ANOVA-style decomposition.

For each topology, fit the landscape f(x) = μ + Σ α_i(x_i) + Σ β_ij(x_i, x_j) + residual.
Applied to log circuit score (m_logic) and negative log growth (m_burden) using the
labeled training-data set, and separately to the MLP-predicted values.

The α_i are slot main effects (one-hot per (slot, part)); β_ij are slot-pair
interaction effects (one-hot per (slot_pair, part_pair)). We fit by OLS with a
zero-sum identifiability constraint handled implicitly via `drop_first` for each
slot's indicator block.

Outputs:
  processed/decomp_{name}_{target}.json        per-slot main-effect table + residual R²
  figures/decomp_{name}.png                    variance-decomp bar + slot × part heatmap
  reports/global_decomposition.md              narrative across 3 exemplars

For interpretability we report:
  - variance explained by μ only (0), μ+α (main-effect model), μ+α+β (pairwise model)
  - largest positive / largest negative α_i(x_i) for each target
  - strongest β_ij(x_i, x_j) pairs
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loaders as L
from loaders import display_name
from plotting_utils import apply_exploratory_style, save_figure_with_data

apply_exploratory_style()

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "processed"
FIGS = ROOT / "figures"
REPORTS = ROOT / "reports"
FIGS.mkdir(parents=True, exist_ok=True)


def onehot_main(perms: np.ndarray, g: int = L.LIBRARY_SIZE) -> np.ndarray:
    """(N, l) -> (N, l*g) one-hot."""
    N, l = perms.shape
    X = np.zeros((N, l * g), dtype=np.float32)
    for i in range(N):
        for s in range(l):
            X[i, s * g + perms[i, s]] = 1.0
    return X


def onehot_pair(perms: np.ndarray, g: int = L.LIBRARY_SIZE) -> np.ndarray:
    """(N, l) -> (N, C(l,2)*g*g) one-hot of slot-pair (part, part)."""
    N, l = perms.shape
    n_pairs = l * (l - 1) // 2
    X = np.zeros((N, n_pairs * g * g), dtype=np.float32)
    for i in range(N):
        col = 0
        for s1, s2 in combinations(range(l), 2):
            X[i, col + perms[i, s1] * g + perms[i, s2]] = 1.0
            col += g * g
    return X


def r2_score(y, yhat):
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def decompose(name: str, n_sample: int = 60000) -> dict:
    td = L.load_training_data(name)
    # subsample for tractability — pairwise design matrix grows as C(l,2) * g²
    if len(td) > n_sample:
        td = td.sample(n_sample, random_state=0).reset_index(drop=True)
    perms = np.stack([np.asarray(p, dtype=np.int64) for p in td["permutation"]])
    l = perms.shape[1]
    y_logic = np.log(np.clip(td.circuit_score.values, 1e-6, None))
    y_burden = -np.log(np.clip(td.growth_score.values, 1e-6, None))

    out = {"n": len(td), "l": l, "g": L.LIBRARY_SIZE, "targets": {}}

    # Slot labels for interpretation
    slots = L.regulator_slot_metadata(name)

    X_main = onehot_main(perms)
    X_pair = onehot_pair(perms)

    for tgt_name, y in [("m_logic", y_logic), ("m_burden", y_burden)]:
        # Null model: just mean
        y_null = np.full_like(y, y.mean())
        # Main-effect model: μ + α
        main = Ridge(alpha=1.0, fit_intercept=True)
        main.fit(X_main, y)
        y_main = main.predict(X_main)
        # Pairwise model: μ + α + β
        full = Ridge(alpha=1.0, fit_intercept=True)
        full.fit(np.hstack([X_main, X_pair]), y)
        y_full = full.predict(np.hstack([X_main, X_pair]))

        alpha_vec = main.coef_.reshape(l, L.LIBRARY_SIZE)
        # top 5 positive/negative (slot, part) by main effect
        top_pos = []; top_neg = []
        flat = [(s, p, float(alpha_vec[s, p])) for s in range(l) for p in range(L.LIBRARY_SIZE)]
        flat_sorted = sorted(flat, key=lambda t: -t[2])
        top_pos = flat_sorted[:5]; top_neg = flat_sorted[-5:]

        # variance decomposition
        out["targets"][tgt_name] = {
            "R2_null": r2_score(y, y_null),
            "R2_main": r2_score(y, y_main),
            "R2_full": r2_score(y, y_full),
            "alpha": alpha_vec.tolist(),
            "top_positive": [{"slot": int(s), "slot_label": f"{slots.iloc[s].repressor}-{slots.iloc[s].rbs}",
                              "part_idx": int(p), "alpha": a} for s, p, a in top_pos],
            "top_negative": [{"slot": int(s), "slot_label": f"{slots.iloc[s].repressor}-{slots.iloc[s].rbs}",
                              "part_idx": int(p), "alpha": a} for s, p, a in top_neg],
        }
    with open(PROC / f"decomp_{name}.json", "w") as f:
        json.dump(out, f, indent=2)

    # Figure
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for col, tgt_name in enumerate(["m_logic", "m_burden"]):
        t = out["targets"][tgt_name]
        alpha = np.array(t["alpha"])
        ax = axes[0, col]
        im = ax.imshow(alpha, aspect="auto", cmap="RdBu_r",
                       vmax=np.abs(alpha).max(), vmin=-np.abs(alpha).max())
        ax.set_yticks(range(l))
        ax.set_yticklabels([f"{slots.iloc[i].repressor}-{slots.iloc[i].rbs}" for i in range(l)], fontsize=7)
        ax.set_xlabel("part index (0..19)")
        ax.set_title(f"{display_name(name)} {tgt_name} main-effect α_i(x_i)")
        plt.colorbar(im, ax=ax)

        ax = axes[1, col]
        # variance explained bars
        labels = ["null (μ)", "μ+α (main)", "μ+α+β (pair)"]
        vals = [0, t["R2_main"], t["R2_full"]]
        ax.bar(labels, vals)
        ax.set_ylim(0, 1)
        ax.set_ylabel("R² on training subsample")
        ax.set_title(f"{display_name(name)} {tgt_name} variance explained")
        for i, v in enumerate(vals):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    save_figure_with_data(
        fig, FIGS / f"decomp_{name}",
        source_files=[
            (f"processed/decomp_{name}.json",
             "ANOVA α / β decomposition at library-index resolution"),
        ],
        how_to_replot="variance bars + slot×part α heatmap from the JSON",
        script_name="06_global_decomposition.py", dpi=140,
    )
    plt.close(fig)
    return out


def main():
    with open(REPORTS / "global_decomposition.md", "w") as f:
        f.write("# Module D — Global decomposition\n\n")
        f.write("Ridge OLS fit of f(x) = μ + Σα_i(x_i) + Σβ_ij(x_i,x_j) + residual\n")
        f.write("on log-circuit and −log-growth targets, per topology.\n\n")
        for name in L.EXEMPLARS:
            print(f"[{name}] decomposing ...")
            out = decompose(name)
            f.write(f"## {name}\n\n")
            f.write(f"- n={out['n']}, l={out['l']}, g={out['g']}\n\n")
            for tgt in ["m_logic", "m_burden"]:
                t = out["targets"][tgt]
                f.write(f"### target = {tgt}\n")
                f.write(f"- R²(main) = {t['R2_main']:.3f}, R²(main+pair) = {t['R2_full']:.3f}\n")
                f.write(f"- Top positive main effects:\n")
                for x in t["top_positive"]:
                    f.write(f"  - slot {x['slot']} ({x['slot_label']}), part {x['part_idx']}: α={x['alpha']:+.3f}\n")
                f.write(f"- Top negative main effects:\n")
                for x in t["top_negative"]:
                    f.write(f"  - slot {x['slot']} ({x['slot_label']}), part {x['part_idx']}: α={x['alpha']:+.3f}\n")
                f.write("\n")
            f.write(f"Figure: `figures/decomp_{name}.png`\n\n")


if __name__ == "__main__":
    main()
