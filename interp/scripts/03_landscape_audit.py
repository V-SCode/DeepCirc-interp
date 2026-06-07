#!/usr/bin/env python3
"""Module A — Reconstruct Fig. 3b landscapes and run a faithfulness audit.

Focus: does the MLP rank designs correctly in the region that matters for
selection (top basin, feasible Pareto front, yellow-dot neighborhood)?
Summary R² is reported by the paper; we add top-k recall, Pareto-front recall,
and yellow-dot-rank regret.

Inputs : processed/master_{name}.parquet (from 02_make_master_tables.py).
Outputs: figures/landscape_{name}_*.png   (predicted-vs-actual, Pareto, top-k)
         reports/landscape_audit.md        (human-readable summary across 3 exemplars)
         processed/audit_{name}.json       (machine-readable metrics per exemplar)

This script is fully enabled by existing data — no simulator required.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def _pareto_mask(pts: np.ndarray, maximize=True) -> np.ndarray:
    """Return bool mask of Pareto-optimal rows. pts shape (N, k)."""
    pts = np.asarray(pts)
    if not maximize:
        pts = -pts
    is_dom = np.zeros(len(pts), dtype=bool)
    # Sort by first obj descending, then scan
    order = np.argsort(-pts[:, 0])
    best_second = -np.inf
    for i in order:
        if pts[i, 1] > best_second:
            best_second = pts[i, 1]
        else:
            is_dom[i] = True
    return ~is_dom


def audit_one(name: str) -> dict:
    m = pd.read_parquet(PROC / f"master_{name}.parquet")

    # Use rows where we have sim labels (training OR nn_high)
    labeled = m[m.circuit_score_sim.notna()].copy()
    labeled["m_logic_sim"] = np.log(np.clip(labeled.circuit_score_sim, 1e-12, None))
    labeled["m_logic_pred"] = np.log(np.clip(labeled.circuit_score_pred, 1e-12, None))
    labeled["m_burden_sim"] = -np.log(np.clip(labeled.growth_score_sim, 1e-12, None))
    labeled["m_burden_pred"] = -np.log(np.clip(labeled.growth_score_pred, 1e-12, None))

    # --- metrics ---
    def r2(a, b):
        a = np.asarray(a); b = np.asarray(b)
        ss_res = float(((a - b) ** 2).sum())
        ss_tot = float(((a - a.mean()) ** 2).sum())
        return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    def spearman(a, b):
        return float(pd.Series(a).corr(pd.Series(b), method="spearman"))

    metrics = {
        "n_labeled": int(len(labeled)),
        "circuit_R2_raw": r2(labeled.circuit_score_sim, labeled.circuit_score_pred),
        "circuit_R2_log": r2(labeled.m_logic_sim, labeled.m_logic_pred),
        "circuit_spearman": spearman(labeled.circuit_score_sim, labeled.circuit_score_pred),
        "growth_R2_raw": r2(labeled.growth_score_sim, labeled.growth_score_pred),
        "growth_R2_log": r2(labeled.m_burden_sim, labeled.m_burden_pred),
        "growth_spearman": spearman(labeled.growth_score_sim, labeled.growth_score_pred),
    }

    # top-k recall: of top-K designs by sim circuit-score, how many does pred rank in top-K?
    for K in [10, 50, 100, 500]:
        if len(labeled) < K:
            continue
        top_sim = set(labeled.nlargest(K, "circuit_score_sim")["perm_str"])
        top_pred = set(labeled.nlargest(K, "circuit_score_pred")["perm_str"])
        metrics[f"circuit_topk_recall_K={K}"] = len(top_sim & top_pred) / K

    # Pareto recall (sim vs pred Pareto front over (circuit_score, growth_score))
    sim_mask = _pareto_mask(labeled[["circuit_score_sim", "growth_score_sim"]].to_numpy())
    pred_mask = _pareto_mask(labeled[["circuit_score_pred", "growth_score_pred"]].to_numpy())
    sim_front = set(labeled[sim_mask]["perm_str"])
    pred_front = set(labeled[pred_mask]["perm_str"])
    metrics["pareto_sim_n"] = int(sim_mask.sum())
    metrics["pareto_pred_n"] = int(pred_mask.sum())
    metrics["pareto_recall"] = len(sim_front & pred_front) / max(1, len(sim_front))

    # yellow-dot regret: where does the sim-best land in the pred ranking, and vice versa?
    yd = L.get_yellow_dot_info(name)
    yd_pstr = "-".join(str(int(x)) for x in yd["perm"])
    pred_rank_of_yd = int((labeled["circuit_score_pred"] > labeled.loc[labeled.perm_str == yd_pstr,
                                                                       "circuit_score_pred"].iloc[0]).sum() + 1)
    metrics["yd_sim_c"] = yd["circuit_score"]
    metrics["yd_pred_c"] = float(labeled.loc[labeled.perm_str == yd_pstr, "circuit_score_pred"].iloc[0])
    metrics["yd_pred_rank_by_circuit"] = pred_rank_of_yd

    # --- figures ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    # (0,0) pred vs actual circuit
    ax = axes[0, 0]
    ax.scatter(labeled.circuit_score_sim, labeled.circuit_score_pred, s=2, alpha=0.15)
    lim = max(labeled.circuit_score_sim.max(), labeled.circuit_score_pred.max())
    ax.plot([0, lim], [0, lim], "k--", lw=0.7)
    ax.set_xlabel("circuit score (sim)"); ax.set_ylabel("circuit score (pred)")
    ax.set_title(f"{display_name(name)} circuit pred vs sim (R²={metrics['circuit_R2_raw']:.3f})")
    # (0,1) pred vs actual growth
    ax = axes[0, 1]
    ax.scatter(labeled.growth_score_sim, labeled.growth_score_pred, s=2, alpha=0.15)
    ax.plot([0, 1], [0, 1], "k--", lw=0.7)
    ax.set_xlabel("growth score (sim)"); ax.set_ylabel("growth score (pred)")
    ax.set_title(f"{display_name(name)} growth pred vs sim (R²={metrics['growth_R2_raw']:.3f})")
    # (0,2) Pareto
    ax = axes[0, 2]
    ax.scatter(labeled.circuit_score_sim, labeled.growth_score_sim, s=1, alpha=0.1, color="gray", label="all")
    pts = labeled.loc[sim_mask].sort_values("circuit_score_sim")
    ax.plot(pts.circuit_score_sim, pts.growth_score_sim, "b.-", ms=3, label=f"sim Pareto ({sim_mask.sum()})")
    pts = labeled.loc[pred_mask].sort_values("circuit_score_pred")
    ax.plot(pts.circuit_score_pred, pts.growth_score_pred, "r.-", ms=3, alpha=0.4,
            label=f"pred Pareto ({pred_mask.sum()})")
    ax.scatter([yd["circuit_score"]], [yd["growth_score"]], s=80, marker="*", c="gold",
               edgecolor="k", zorder=10, label="yellow dot")
    ax.set_xlabel("circuit score"); ax.set_ylabel("growth score")
    ax.set_title(f"{display_name(name)} Pareto front (recall={metrics['pareto_recall']:.3f})")
    ax.legend(fontsize=7)
    # (1,0) top-k recall
    ax = axes[1, 0]
    ks = [K for K in [10, 50, 100, 500] if f"circuit_topk_recall_K={K}" in metrics]
    ax.plot(ks, [metrics[f"circuit_topk_recall_K={K}"] for K in ks], "o-")
    ax.set_xlabel("K"); ax.set_ylabel("top-K sim ∩ top-K pred / K")
    ax.set_title(f"{display_name(name)} top-K recall (circuit)"); ax.set_ylim(0, 1.05)
    # (1,1) hist of log margins
    ax = axes[1, 1]
    ax.hist(labeled.m_logic_sim, bins=60, alpha=0.6, label="log c sim", density=True)
    ax.hist(labeled.m_logic_pred, bins=60, alpha=0.4, label="log c pred", density=True)
    ax.axvline(0, color="k", ls="--", lw=0.5)
    ax.set_xlabel("log circuit score"); ax.set_ylabel("density")
    ax.legend(fontsize=7); ax.set_title(f"{display_name(name)} m_logic = log(c)")
    # (1,2) yellow-dot rank info
    ax = axes[1, 2]
    ranks_by_sim = np.argsort(-labeled.circuit_score_sim.values)
    ranks_by_pred = np.argsort(-labeled.circuit_score_pred.values)
    sim_ranks = np.argsort(ranks_by_sim)
    pred_ranks = np.argsort(ranks_by_pred)
    yd_idx = labeled.index[labeled.perm_str == yd_pstr][0]
    yd_row = labeled.index.get_loc(yd_idx)
    top1k = np.argsort(-labeled.circuit_score_sim.values)[:min(1000, len(labeled))]
    ax.scatter(sim_ranks[top1k], pred_ranks[top1k], s=2, alpha=0.3)
    ax.plot([0, len(labeled)], [0, len(labeled)], "k--", lw=0.5)
    ax.scatter([sim_ranks[yd_row]], [pred_ranks[yd_row]], s=80, marker="*", c="gold",
               edgecolor="k", zorder=10)
    ax.set_xlabel("rank by sim c"); ax.set_ylabel("rank by pred c")
    ax.set_title(f"{display_name(name)} rank agreement (top-1k sim)")

    fig.suptitle(f"Landscape audit — {name}  (n_labeled={len(labeled):,})")
    fig.tight_layout()
    save_figure_with_data(
        fig, FIGS / f"landscape_{name}",
        source_files=[
            (f"processed/audit_{name}.json",
             "per-exemplar faithfulness-audit metrics"),
            (f"processed/master_{name}.parquet",
             "merged sim+MLP labels per design"),
        ],
        how_to_replot="recompute from master parquet (MLP pred vs sim label scatter "
                      "+ rank density); audit metrics in audit_{name}.json",
        script_name="03_landscape_audit.py", dpi=140,
    )
    plt.close(fig)

    with open(PROC / f"audit_{name}.json", "w") as f:
        json.dump({k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()},
                  f, indent=2, default=str)

    return metrics


def main():
    rows = []
    for name in L.EXEMPLARS:
        print(f"[{name}] auditing ...")
        m = audit_one(name)
        rows.append({"name": name, **m})

    # Human report
    with open(REPORTS / "landscape_audit.md", "w") as f:
        f.write("# Module A — Landscape audit\n\n")
        f.write("Faithfulness of the MLP predictions against simulator labels,\n")
        f.write("with emphasis on the region that matters for selection.\n\n")
        f.write("All metrics computed on the union of training-set + NN-predicted-high rows\n")
        f.write("(the rows where we have simulator ground truth).\n\n")
        f.write("| metric | 0x2B | 0x17 | 0x6D |\n|---|---|---|---|\n")
        keys = list(rows[0].keys()); keys.remove("name")
        for k in keys:
            vals = [rows[i].get(k, "") for i in range(len(rows))]
            f.write(f"| `{k}` | " + " | ".join(
                f"{v:.4g}" if isinstance(v, float) else str(v) for v in vals) + " |\n")
        f.write("\nFigures: `figures/landscape_{0x2B,0x17,0x6D}.png`.\n")
    print(f"\nwrote {REPORTS/'landscape_audit.md'}")
    print(f"figures: {sorted(FIGS.glob('landscape_*.png'))}")


if __name__ == "__main__":
    main()
