#!/usr/bin/env python3
"""Module C — Controlled good-vs-bad comparisons.

For each topology, select:
  - logic-failure comparator: similar growth, much lower circuit score
  - growth-failure comparator: similar circuit, much lower growth score
  - interference comparator: NN predicted high but flagged for interference
  - near-miss comparator: close to yellow dot in Hamming distance, lower score

Per comparison we record the slot diffs, score diffs, and each slot's
contribution to the gap. Detailed per-state "weakest ON / leakiest OFF"
analysis requires the simulator per-state outputs — those are NOT in our
folders. This script flags that gap explicitly.

Outputs:
  processed/comparisons_{name}.parquet
  figures/comparisons_{name}.png
  reports/good_bad_comparisons.md
"""
from __future__ import annotations

import sys
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


def hamming(a: tuple, b: tuple) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


def select_comparators(name: str) -> pd.DataFrame:
    m = pd.read_parquet(PROC / f"master_{name}.parquet")
    yd = L.get_yellow_dot_info(name)
    yd_perm = tuple(int(x) for x in yd["perm"])
    yd_c = yd["circuit_score"]; yd_g = yd["growth_score"]

    # Only labeled, non-interference rows are candidates for non-interference comparators
    labeled = m[m.circuit_score_sim.notna()].copy()
    labeled["permutation"] = labeled["permutation"].apply(tuple)
    labeled["hamming_to_yd"] = labeled["permutation"].apply(lambda p: hamming(p, yd_perm))

    # Non-interference pool
    ni = labeled[~labeled.interference_flag_sim].copy()

    comps = []

    # Yellow dot itself (reference)
    yd_row = labeled[labeled.perm_str == "-".join(str(x) for x in yd_perm)].iloc[0].to_dict()
    comps.append({"role": "yellow_dot", **yd_row})

    if len(ni) > 50:
        # logic-failure: growth within ±5% of yd_g, much lower circuit score (bottom 10%)
        tol_g = 0.05 * yd_g
        close_g = ni[(ni.growth_score_sim - yd_g).abs() <= tol_g]
        if len(close_g) > 0:
            r = close_g.nsmallest(1, "circuit_score_sim").iloc[0].to_dict()
            comps.append({"role": "logic_failure", **r})

        # growth-failure: circuit within ±20% of yd_c, much lower growth
        tol_c = 0.20 * yd_c
        close_c = ni[(ni.circuit_score_sim - yd_c).abs() <= tol_c]
        if len(close_c) > 0:
            r = close_c.nsmallest(1, "growth_score_sim").iloc[0].to_dict()
            comps.append({"role": "growth_failure", **r})

        # near-miss: small Hamming distance, lower circuit than yd
        nm = ni[(ni.hamming_to_yd >= 1) & (ni.hamming_to_yd <= 2)
                & (ni.circuit_score_sim < yd_c * 0.7)]
        if len(nm) > 0:
            r = nm.nlargest(1, "circuit_score_sim").iloc[0].to_dict()
            comps.append({"role": "near_miss", **r})

    # interference comparator: high predicted c, but interference flag true
    inter = labeled[(labeled.interference_flag_sim) & (labeled.circuit_score_pred > yd_c * 0.5)]
    if len(inter) > 0:
        r = inter.nlargest(1, "circuit_score_pred").iloc[0].to_dict()
        comps.append({"role": "interference", **r})

    # best-in-training comparator
    try:
        top_train = L.load_top_in_training(name).iloc[0]
        tp = tuple(int(x) for x in top_train.permutation)
        row = labeled[labeled.perm_str == "-".join(str(x) for x in tp)]
        if len(row) > 0:
            comps.append({"role": "top_in_training", **row.iloc[0].to_dict()})
    except Exception as e:
        print(f"  [{name}] top_in_training unavailable: {e}")

    df = pd.DataFrame(comps)
    # Slot diffs vs yellow dot
    yd_p = list(yd_perm)
    def diff_vec(row):
        p = tuple(int(x) for x in row.permutation)
        return [int(p[s] != yd_p[s]) for s in range(len(yd_p))]
    df["slot_diff_mask"] = df.apply(diff_vec, axis=1)
    df["n_slot_diffs"] = df["slot_diff_mask"].apply(sum)
    return df


def plot_one(name: str, df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    roles = df["role"].tolist()
    cs = df["circuit_score_sim"].astype(float).values
    gs = df["growth_score_sim"].astype(float).values
    for i, r in enumerate(roles):
        ax.scatter(cs[i], gs[i], s=90, label=r)
        ax.annotate(r, (cs[i], gs[i]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("circuit score (sim)"); ax.set_ylabel("growth score (sim)")
    ax.set_title(f"{name} matched comparators")
    ax.grid(alpha=0.3)

    ax = axes[1]
    slots = L.regulator_slot_metadata(name)
    labels = [f"s{r.slot_idx} ({r.repressor}-{r.rbs})" for _, r in slots.iterrows()]
    mat = np.stack([np.array(row) for row in df["slot_diff_mask"]])
    ax.imshow(mat, aspect="auto", cmap="gray_r")
    ax.set_yticks(range(len(df))); ax.set_yticklabels(df["role"])
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_title(f"{name} slot-difference mask (dark = differs from yd)")
    fig.tight_layout()
    save_figure_with_data(
        fig, FIGS / f"comparisons_{name}",
        source_files=[
            (f"processed/comparisons_{name}.parquet",
             "per-comparator score-level table (role, perm, c, g, slot diffs)"),
        ],
        how_to_replot="score bars + slot-diff heatmap from the parquet",
        script_name="05_good_bad_comparisons.py", dpi=140,
    )
    plt.close(fig)


def main():
    with open(REPORTS / "good_bad_comparisons.md", "w") as f:
        f.write("# Module C — Controlled good-vs-bad comparisons\n\n")
        f.write("Matched comparators selected per exemplar. All metrics are from the\n")
        f.write("simulator-labeled subset. Per-state (weakest-ON / leakiest-OFF) and\n")
        f.write("per-gate-bottleneck analyses require simulator reimplementation and\n")
        f.write("are **not produced here** — see `missing_data_request.md`.\n\n")
        for name in L.EXEMPLARS:
            print(f"[{name}] selecting comparators ...")
            df = select_comparators(name)
            df.to_parquet(PROC / f"comparisons_{name}.parquet", index=False)
            plot_one(name, df)
            f.write(f"## {name}\n\n")
            cols = ["role", "permutation", "circuit_score_sim", "growth_score_sim",
                    "interference_flag_sim", "circuit_score_pred", "n_slot_diffs"]
            avail = [c for c in cols if c in df.columns]
            f.write(df[avail].to_markdown(index=False, floatfmt=".4f") + "\n\n")
            f.write(f"Figure: `figures/comparisons_{name}.png`\n\n")
    print(f"wrote {REPORTS/'good_bad_comparisons.md'}")


if __name__ == "__main__":
    main()
