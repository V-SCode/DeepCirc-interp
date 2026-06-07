#!/usr/bin/env python3
"""Module G — Stagewise decision-pipeline audit (funnel).

Reconstruct the DeepCirc selection funnel per exemplar:
  Stage 1: all labeled assignments in the training-data pickle
  Stage 2: NN predicted-high threshold pass (this set = "scored_high_predicted")
  Stage 3: promoter-interference filter (interference_flag_sim = False)
  Stage 4: growth-threshold pass (growth_score_sim >= 0.75 for l<=6, else 0.65)
  Stage 5: final top-k by simulator circuit score

Outputs:
  processed/funnel_{name}.json
  figures/funnel_{name}.png
  reports/stagewise_audit.md
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


def audit(name: str) -> dict:
    ex = L.EXEMPLARS[name]
    m = pd.read_parquet(PROC / f"master_{name}.parquet")

    growth_thresh = 0.75 if ex.n_regs <= 6 else 0.65

    stages = {}
    # Stage 1: all labeled (training union nn_high)
    labeled = m[m.circuit_score_sim.notna()].copy()
    stages["1_all_labeled"] = len(labeled)

    # Stage 2: NN-picked high (source_nn_high == True)
    nn_high = labeled[labeled.source_nn_high]
    stages["2_nn_high"] = len(nn_high)

    # Stage 3: non-interference
    non_int = nn_high[~nn_high.interference_flag_sim]
    stages["3_non_interference"] = len(non_int)

    # Stage 4: growth threshold
    pass_g = non_int[non_int.growth_score_sim >= growth_thresh]
    stages["4_growth_thresh"] = len(pass_g)

    # Stage 5: top-ranked in top_ranked CSV
    stages["5_top_ranked"] = int(m.source_top_ranked.sum())

    # First-rejection-reason per nn_high row
    nn_high = nn_high.copy()
    nn_high["reject_reason"] = "survived"
    nn_high.loc[nn_high.interference_flag_sim, "reject_reason"] = "interference"
    mask = (~nn_high.interference_flag_sim) & (nn_high.growth_score_sim < growth_thresh)
    nn_high.loc[mask, "reject_reason"] = "growth<thresh"

    reject_tab = nn_high.reject_reason.value_counts().to_dict()

    # Save
    out = {
        "name": name,
        "growth_threshold": growth_thresh,
        "stages": stages,
        "reject_reasons_among_nn_high": reject_tab,
    }
    with open(PROC / f"funnel_{name}.json", "w") as f:
        json.dump(out, f, indent=2)

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    keys = ["1_all_labeled", "2_nn_high", "3_non_interference", "4_growth_thresh", "5_top_ranked"]
    vals = [stages[k] for k in keys]
    ax.bar(keys, vals)
    ax.set_yscale("log")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8, rotation=0)
    ax.set_ylabel("# assignments (log)"); ax.set_title(f"{display_name(name)} funnel")
    ax.tick_params(axis="x", labelrotation=30)

    ax = axes[1]
    colors = {"interference": "#c33", "growth<thresh": "#e8b", "survived": "#3a7"}
    sc = ax.scatter(nn_high.circuit_score_sim, nn_high.growth_score_sim,
                    c=nn_high.reject_reason.map(lambda r: colors.get(r, "#888")),
                    s=3, alpha=0.4)
    ax.axhline(growth_thresh, color="k", ls="--", lw=0.5)
    ax.set_xlabel("circuit score (sim)"); ax.set_ylabel("growth score (sim)")
    ax.set_title(f"{display_name(name)} NN-high: first rejection reason")
    for r, c in colors.items():
        ax.scatter([], [], c=c, label=r, s=20)
    ax.legend(fontsize=7)
    fig.tight_layout()
    save_figure_with_data(
        fig, FIGS / f"funnel_{name}",
        source_files=[
            (f"processed/funnel_{name}.json",
             "stagewise counts + rejection-reason distribution"),
        ],
        how_to_replot="funnel bar chart + first-rejection scatter from the JSON",
        script_name="07_stagewise_audit.py", dpi=140,
    )
    plt.close(fig)
    return out


def main():
    with open(REPORTS / "stagewise_audit.md", "w") as f:
        f.write("# Module G — Stagewise decision-pipeline audit\n\n")
        f.write("| stage | 0x2B | 0x17 | 0x6D |\n|---|---|---|---|\n")
        outs = []
        for name in L.EXEMPLARS:
            print(f"[{name}] auditing funnel ...")
            o = audit(name); outs.append(o)
        keys = list(outs[0]["stages"].keys())
        for k in keys:
            vals = [outs[i]["stages"].get(k, "") for i in range(len(outs))]
            f.write(f"| `{k}` | " + " | ".join(f"{v:,}" for v in vals) + " |\n")
        f.write("\n")
        for o in outs:
            f.write(f"## {o['name']}\n")
            f.write(f"- Growth threshold applied: {o['growth_threshold']}\n")
            f.write(f"- Rejection reasons among NN-high: {o['reject_reasons_among_nn_high']}\n")
            f.write(f"- Figure: `figures/funnel_{o['name']}.png`\n\n")
    print(f"wrote {REPORTS/'stagewise_audit.md'}")


if __name__ == "__main__":
    main()
