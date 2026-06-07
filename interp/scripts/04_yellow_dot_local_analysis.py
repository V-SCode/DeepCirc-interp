#!/usr/bin/env python3
"""Module B — Local yellow-dot explanation.

For each yellow-dot assignment, compute:
 - single-slot counterfactual swaps (exhaustive over g=20 parts per slot,
   skipping swaps that would violate no-repeated-protein family orthogonality
   — we approximate by skipping same-index-as-other-slot swaps since the
   index→family mapping is only partially known).
 - double-slot counterfactual swaps (top-N by magnitude on single-slot first).
 - minimal failing edits: for each swap, record whether predicted circuit,
   predicted growth, and simulated (if labeled) change state.
 - on-manifold slot-level Shapley values using the training-data set K as the
   valid completion marginal, with the MLP as the value function.

Constraint: vanilla SHAP on one-hot is WRONG. We use slot-level Shapley with
valid-completion averaging drawn from the training-data permutation set K.

Outputs:
  processed/local_{name}.parquet      (long-format slot/swap records)
  processed/shapley_{name}.json       (slot-Shapley values for circuit & growth)
  figures/yellow_dot_{name}.png       (waterfall + interaction heatmap + subst. map)
  reports/local_yellow_dot.md         (narrative across 3 exemplars)

Simulator-side versions of these analyses require re-simulating novel perms,
which is blocked on POSTECH JSONs + utils5.py. This script uses MLP proxies
for counterfactual values; sim ground truth is reported only where labeled.
"""
from __future__ import annotations

import json
import sys
import math
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


def single_slot_swaps(name: str, perm: tuple, cm, gm):
    """For the yellow-dot perm, enumerate every single-slot swap to each of g parts.
    Skip swaps where the new part already appears in another slot (orthogonality proxy).
    """
    l = len(perm)
    perm = tuple(int(x) for x in perm)
    rows = []
    for s in range(l):
        for new in range(L.LIBRARY_SIZE):
            if new == perm[s]:
                continue
            if new in [perm[k] for k in range(l) if k != s]:
                # would cause index collision; skip (approximates family-orthogonality)
                continue
            new_perm = list(perm)
            new_perm[s] = new
            rows.append({"slot": s, "orig_part": perm[s], "new_part": new,
                         "new_perm": tuple(new_perm)})
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    perms = np.stack([np.asarray(p, dtype=np.int64) for p in df["new_perm"]])
    df["circuit_pred_after"] = L.predict(cm, perms)
    df["growth_pred_after"] = L.predict(gm, perms)
    # baseline predictions at yellow-dot
    base_c = float(L.predict(cm, [list(perm)])[0])
    base_g = float(L.predict(gm, [list(perm)])[0])
    df["delta_circuit_pred"] = df["circuit_pred_after"] - base_c
    df["delta_growth_pred"] = df["growth_pred_after"] - base_g
    df["base_circuit_pred"] = base_c
    df["base_growth_pred"] = base_g
    return df


def double_slot_swaps(name: str, perm: tuple, cm, gm, top_n_per_slot: int = 5):
    """Given single-slot results, pick each slot's top_n single-swap candidates,
    then enumerate all pairs-of-slots-with-candidates as double swaps and score them.
    """
    single = single_slot_swaps(name, perm, cm, gm)
    if len(single) == 0:
        return pd.DataFrame()
    l = len(perm)
    # Pick best |delta_circuit_pred| per slot (either direction)
    by_slot = {}
    for s in range(l):
        sub = single[single.slot == s].copy()
        sub["abs_delta"] = sub["delta_circuit_pred"].abs()
        by_slot[s] = sub.nlargest(top_n_per_slot, "abs_delta")["new_part"].tolist()

    rows = []
    base_c = float(L.predict(cm, [list(perm)])[0])
    base_g = float(L.predict(gm, [list(perm)])[0])
    for s1, s2 in combinations(range(l), 2):
        for p1 in by_slot[s1]:
            for p2 in by_slot[s2]:
                new_perm = list(perm)
                new_perm[s1] = p1; new_perm[s2] = p2
                if len(set(new_perm)) != l:
                    continue
                rows.append({"s1": s1, "s2": s2, "p1": p1, "p2": p2,
                             "new_perm": tuple(new_perm)})
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    perms = np.stack([np.asarray(p, dtype=np.int64) for p in df["new_perm"]])
    df["circuit_pred_after"] = L.predict(cm, perms)
    df["growth_pred_after"] = L.predict(gm, perms)
    df["delta_circuit_pred"] = df["circuit_pred_after"] - base_c
    df["delta_growth_pred"] = df["growth_pred_after"] - base_g
    df["base_circuit_pred"] = base_c
    df["base_growth_pred"] = base_g
    return df


def slot_shapley(perm: tuple, cm, gm, K_perms: np.ndarray,
                 model_circuit=True, n_samples: int = 200, rng=None) -> dict:
    """On-manifold slot-level Shapley estimated by random-coalition sampling.

    Players = regulator slots. Given a coalition S, we fix x_S to the yellow-dot
    values and marginalize the rest by drawing valid completions from K_perms
    (the training-data permutation set). Value function = model prediction.

    Monte-Carlo Shapley with n_samples random permutations of player order.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    l = len(perm)
    perm = np.asarray(perm, dtype=np.int64)
    model = cm if model_circuit else gm
    # Pool of completions
    completions = K_perms  # (N, l)
    N = len(completions)
    shap = np.zeros(l, dtype=np.float64)

    def v_S(S_mask, n_marginal=128):
        # Fix x_S = yellow-dot; sample remaining slots from completions' values in those slots
        # We marginalize by sampling random rows from K and only using their non-S entries.
        idx = rng.integers(0, N, size=n_marginal)
        drawn = completions[idx]               # (n_marginal, l)
        out = drawn.copy()
        out[:, S_mask] = perm[S_mask]
        preds = L.predict(model, out, batch_size=max(n_marginal, 1024))
        return float(preds.mean())

    # v(∅) is the expected model output when NO slots are fixed — the marginal
    # baseline that Shapley values are defined against. Estimate it once with
    # a larger n_marginal since every ordering starts from it.
    v_empty = v_S(np.zeros(l, dtype=bool), n_marginal=512)

    for _ in range(n_samples):
        order = rng.permutation(l)
        S_mask = np.zeros(l, dtype=bool)
        v_prev = v_empty
        for p in order:
            S_mask[p] = True
            v_new = v_S(S_mask, n_marginal=64)
            shap[p] += v_new - v_prev
            v_prev = v_new
    shap /= n_samples
    # `baseline` now stores v(∅); by Shapley efficiency ΣΦᵢ = total_at_yd − baseline.
    return {"shap": shap.tolist(),
            "baseline": float(v_empty),
            "total_at_yd": float(L.predict(model, [perm.tolist()])[0])}


def analyze_one(name: str) -> dict:
    print(f"[{name}] local yellow-dot analysis ...")
    yd = L.get_yellow_dot_info(name)
    perm = tuple(int(x) for x in yd["perm"])
    cm = L.load_circuit_mlp(name)
    gm = L.load_growth_mlp(name)

    # 1. Single-slot swaps
    single = single_slot_swaps(name, perm, cm, gm)
    single["kind"] = "single"
    print(f"  {len(single)} single-slot swaps")

    # 2. Double-slot swaps
    double = double_slot_swaps(name, perm, cm, gm, top_n_per_slot=5)
    double["kind"] = "double"
    print(f"  {len(double)} double-slot swaps")

    # Combine into long format
    single_long = single.rename(columns={"slot": "s1", "orig_part": "p1_orig",
                                          "new_part": "p1"}).assign(s2=pd.NA, p2=pd.NA)
    rows = pd.concat([
        single_long[["kind", "s1", "s2", "p1", "p2", "new_perm",
                     "circuit_pred_after", "growth_pred_after",
                     "delta_circuit_pred", "delta_growth_pred",
                     "base_circuit_pred", "base_growth_pred"]],
        double[["kind", "s1", "s2", "p1", "p2", "new_perm",
                "circuit_pred_after", "growth_pred_after",
                "delta_circuit_pred", "delta_growth_pred",
                "base_circuit_pred", "base_growth_pred"]],
    ], ignore_index=True)
    rows["new_perm"] = rows["new_perm"].apply(list)
    rows.to_parquet(PROC / f"local_{name}.parquet", index=False)

    # 3. Shapley using training-data K as valid-completion pool
    td = L.load_training_data(name)
    K = np.stack([np.asarray(p, dtype=np.int64) for p in td["permutation"].iloc[:20000]])  # cap for speed
    shap_c = slot_shapley(perm, cm, gm, K, model_circuit=True, n_samples=80)
    shap_g = slot_shapley(perm, cm, gm, K, model_circuit=False, n_samples=80)
    with open(PROC / f"shapley_{name}.json", "w") as f:
        json.dump({"perm": list(perm), "circuit": shap_c, "growth": shap_g}, f, indent=2)

    # 4. Figure
    slots = L.regulator_slot_metadata(name)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    # (0,0) Shapley waterfall for circuit
    ax = axes[0, 0]
    labels = [f"s{r.slot_idx}\n{r.repressor}-{r.rbs}" for _, r in slots.iterrows()]
    y = np.array(shap_c["shap"])
    ax.bar(range(len(y)), y, color=["#3a7" if v >= 0 else "#c33" for v in y])
    ax.set_xticks(range(len(y))); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.axhline(0, color="k", lw=0.4)
    ax.set_ylabel("slot Shapley (circuit score)"); ax.set_title(f"{name} slot-Shapley (circuit)")
    # (0,1) Shapley for growth
    ax = axes[0, 1]
    y = np.array(shap_g["shap"])
    ax.bar(range(len(y)), y, color=["#3a7" if v >= 0 else "#c33" for v in y])
    ax.set_xticks(range(len(y))); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.axhline(0, color="k", lw=0.4)
    ax.set_ylabel("slot Shapley (growth score)"); ax.set_title(f"{name} slot-Shapley (growth)")
    # (1,0) single-slot delta scatter (Δc vs Δg)
    ax = axes[1, 0]
    if len(single) > 0:
        sc = ax.scatter(single.delta_circuit_pred, single.delta_growth_pred,
                        c=single.slot, cmap="tab10", s=20, alpha=0.7)
        ax.axhline(0, color="k", lw=0.3); ax.axvline(0, color="k", lw=0.3)
        plt.colorbar(sc, ax=ax, label="slot")
        ax.set_xlabel("Δ circuit pred"); ax.set_ylabel("Δ growth pred")
        ax.set_title(f"{name} single-slot swaps (Δc vs Δg)")
    # (1,1) minimum Δc/Δg edit ladder
    ax = axes[1, 1]
    if len(single) > 0:
        # For each single-slot swap: compute edit "magnitude" in (Δm_logic, Δm_burden) L2
        dc = single.delta_circuit_pred.values
        dg = single.delta_growth_pred.values
        mag = np.sqrt(dc**2 + (dg * 100) ** 2)  # weight growth more since |Δg| is small
        order = np.argsort(mag)
        ax.plot(mag[order], "o-", ms=3)
        ax.set_xlabel("single-swap rank (by combined Δ magnitude)")
        ax.set_ylabel("combined |Δ|")
        ax.set_title(f"{name} edit-ladder (closest neighbors first)")
    fig.suptitle(f"Yellow-dot local analysis — {name}  "
                 f"(perm={list(perm)}, c_sim={yd['circuit_score']:.2f}, g_sim={yd['growth_score']:.3f})")
    fig.tight_layout()
    save_figure_with_data(
        fig, FIGS / f"yellow_dot_{name}",
        source_files=[
            (f"processed/shapley_{name}.json",
             "main slot-Shapley (MLP-valued), baseline=v(∅), total_at_yd"),
            (f"processed/local_{name}.parquet",
             "single- and double-slot counterfactual swap table"),
        ],
        how_to_replot="waterfall from shapley JSON; scatter/ladder from the parquet",
        script_name="04_yellow_dot_local_analysis.py", dpi=140,
    )
    plt.close(fig)

    return {"name": name,
            "yellow_dot_perm": list(perm),
            "n_single_swaps": int(len(single)),
            "n_double_swaps": int(len(double)),
            "shap_circuit": shap_c["shap"],
            "shap_growth": shap_g["shap"]}


def main():
    summary = []
    for name in L.EXEMPLARS:
        summary.append(analyze_one(name))

    with open(REPORTS / "local_yellow_dot.md", "w") as f:
        f.write("# Module B — Local yellow-dot analysis\n\n")
        f.write("On-manifold slot-level Shapley values + exhaustive single-slot ")
        f.write("and top-ranked double-slot counterfactuals (MLP-valued).\n\n")
        for s in summary:
            slots = L.regulator_slot_metadata(s["name"])
            f.write(f"## {s['name']}\n\n")
            f.write(f"- Yellow-dot perm: `{s['yellow_dot_perm']}`\n")
            f.write(f"- Slots (in perm order):\n\n")
            for _, r in slots.iterrows():
                cc = s["shap_circuit"][r.slot_idx]
                gg = s["shap_growth"][r.slot_idx]
                f.write(f"  - slot {r.slot_idx} ({r.repressor}-{r.rbs}): "
                        f"Shapley c={cc:+.3f}, g={gg:+.4f}\n")
            f.write(f"\nCounterfactuals: `processed/local_{s['name']}.parquet` "
                    f"({s['n_single_swaps']} single + {s['n_double_swaps']} double)\n")
            f.write(f"Figure: `figures/yellow_dot_{s['name']}.png`\n\n")


if __name__ == "__main__":
    main()
