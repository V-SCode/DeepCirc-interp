#!/usr/bin/env python3
"""Module F — Model-internal decision analysis.

Layerwise linear probes predicting:
  - circuit_score_sim (and m_logic)
  - growth_score_sim (and m_burden)
  - interference_flag_sim
  - is yellow-dot neighborhood (Hamming ≤ 2 from yd)

Probed on activations from each Linear-layer output (post-ReLU for hidden layers).

Outputs:
  processed/probes_{name}.json                per-layer probe R² / AUROC
  figures/probes_{name}.png                   layerwise probe bar chart
  reports/model_internal_analysis.md          narrative across 3 exemplars
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loaders as L
from plotting_utils import apply_exploratory_style, save_figure_with_data

apply_exploratory_style()

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "processed"
FIGS = ROOT / "figures"
REPORTS = ROOT / "reports"
FIGS.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def extract_activations(model: nn.Sequential, X: torch.Tensor):
    """Return list of activations [input, after-L1, after-L2, ..., after-Lk]."""
    acts = [X]
    h = X
    for mod in model:
        h = mod(h)
        acts.append(h.clone())
    # keep only post-Linear activations (every Linear output, regardless of ReLU after)
    linear_outs = [acts[0]]
    i = 0
    for mod in model:
        i += 1
        if isinstance(mod, nn.Linear):
            linear_outs.append(acts[i])
    return [a.detach().numpy() for a in linear_outs]


def probe_one(name: str, n_samples: int = 20000) -> dict:
    print(f"[{name}] extracting activations ...")
    cm = L.load_circuit_mlp(name)
    gm = L.load_growth_mlp(name)
    td = L.load_training_data(name)
    if len(td) > n_samples:
        td = td.sample(n_samples, random_state=0).reset_index(drop=True)
    perms = np.stack([np.asarray(p, dtype=np.int64) for p in td["permutation"]])
    X = L.one_hot_batch(perms)
    yd = L.get_yellow_dot_info(name)
    yd_p = np.asarray(yd["perm"], dtype=np.int64)
    hamming = (perms != yd_p).sum(axis=1)
    is_neighborhood = (hamming <= 2).astype(int)

    # Targets
    y_c = td.circuit_score.values.astype(np.float32)
    y_g = td.growth_score.values.astype(np.float32)
    y_mlog = np.log(np.clip(y_c, 1e-6, None))
    y_mbur = -np.log(np.clip(y_g, 1e-6, None))
    y_int = td.interference_flag.astype(int).values

    # train/test 80/20
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(td))
    split = int(0.8 * len(td))
    tr, te = idx[:split], idx[split:]

    results = {}
    for model_tag, model, primary in [("circuit_mlp", cm, "m_logic"),
                                       ("growth_mlp", gm, "m_burden")]:
        acts = extract_activations(model, X)  # list of (N, d_ℓ)
        per_layer = []
        for ℓ, A in enumerate(acts):
            layer_metrics = {"layer": ℓ, "dim": int(A.shape[1])}
            for tgt_name, y in [("m_logic", y_mlog), ("m_burden", y_mbur)]:
                r = Ridge(alpha=1.0)
                r.fit(A[tr], y[tr])
                yh = r.predict(A[te])
                layer_metrics[f"R2_{tgt_name}"] = float(r2_score(y[te], yh))
            # interference AUROC
            try:
                c = LogisticRegression(max_iter=500, n_jobs=1)
                c.fit(A[tr], y_int[tr])
                p = c.predict_proba(A[te])[:, 1]
                layer_metrics["AUROC_interference"] = float(roc_auc_score(y_int[te], p))
            except Exception:
                layer_metrics["AUROC_interference"] = None
            # neighborhood AUROC
            try:
                if is_neighborhood[tr].sum() >= 5 and is_neighborhood[te].sum() >= 5:
                    c = LogisticRegression(max_iter=500, n_jobs=1)
                    c.fit(A[tr], is_neighborhood[tr])
                    p = c.predict_proba(A[te])[:, 1]
                    layer_metrics["AUROC_yd_neighborhood"] = float(roc_auc_score(is_neighborhood[te], p))
                else:
                    layer_metrics["AUROC_yd_neighborhood"] = None
            except Exception:
                layer_metrics["AUROC_yd_neighborhood"] = None
            per_layer.append(layer_metrics)
        results[model_tag] = per_layer

    with open(PROC / f"probes_{name}.json", "w") as f:
        json.dump({"name": name, "n": int(len(td)), **results}, f, indent=2)

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for col, tag in enumerate(["circuit_mlp", "growth_mlp"]):
        ax = axes[col]
        rows = results[tag]
        layers = [r["layer"] for r in rows]
        ax.plot(layers, [r["R2_m_logic"] for r in rows], "o-", label="R² m_logic")
        ax.plot(layers, [r["R2_m_burden"] for r in rows], "s-", label="R² m_burden")
        ia = [r["AUROC_interference"] for r in rows]
        if all(v is not None for v in ia):
            ax.plot(layers, ia, "^-", label="AUROC interference", alpha=0.8)
        na = [r["AUROC_yd_neighborhood"] for r in rows]
        if all(v is not None for v in na):
            ax.plot(layers, na, "v-", label="AUROC yd neighborhood", alpha=0.8)
        ax.set_xlabel("linear-layer index (0=input)")
        ax.set_ylabel("probe metric")
        ax.set_ylim(-0.1, 1.05)
        ax.set_title(f"{name} probes on {tag}")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout()
    save_figure_with_data(
        fig, FIGS / f"probes_{name}",
        source_files=[
            (f"processed/probes_{name}.json",
             "layerwise probe R² / AUROC for both circuit and growth MLPs"),
        ],
        how_to_replot="per-layer R²/AUROC curves from circuit_mlp and growth_mlp keys",
        script_name="08_model_internal_analysis.py", dpi=140,
    )
    plt.close(fig)
    return {"name": name, "results": results}


def main():
    with open(REPORTS / "model_internal_analysis.md", "w") as f:
        f.write("# Module F — Model-internal decision analysis\n\n")
        f.write("Linear probes from each Linear-layer output to:\n")
        f.write("  m_logic (log circuit score), m_burden (-log growth), interference flag,\n")
        f.write("  yellow-dot neighborhood membership (Hamming ≤ 2 from yd).\n\n")
        for name in L.EXEMPLARS:
            o = probe_one(name)
            f.write(f"## {name}\n\n")
            for tag in ["circuit_mlp", "growth_mlp"]:
                f.write(f"### {tag}\n")
                rows = o["results"][tag]
                hdr = ["layer", "dim", "R2_m_logic", "R2_m_burden",
                       "AUROC_interference", "AUROC_yd_neighborhood"]
                f.write("| " + " | ".join(hdr) + " |\n")
                f.write("|" + "|".join(["---"] * len(hdr)) + "|\n")
                for r in rows:
                    vals = [r.get(k, "") for k in hdr]
                    vals = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in vals]
                    f.write("| " + " | ".join(vals) + " |\n")
                f.write("\n")
            f.write(f"Figure: `figures/probes_{name}.png`\n\n")
    print(f"wrote {REPORTS/'model_internal_analysis.md'}")


if __name__ == "__main__":
    main()
