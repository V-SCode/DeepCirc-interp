#!/usr/bin/env python3
"""Module B simulator-valued — main Shapley AND Shapley-Taylor pairwise.

Combines two methodological-caveat upgrades:
  · A1: simulator-valued Shapley-Taylor pair coefficients Φᵢⱼ (currently
        only MLP-valued via 04b).
  · A15: tighter MC variance on main effects — n_marginal 64 → 256.

Both targets (circuit_score, growth_score) come back from a single
simulator pass, so we compute them jointly. Outputs:
  processed/shapley_taylor_sim_{name}.json
    fields: perm, slot_labels, pool_size,
            circuit / growth → {main_shap, pair_shap, baseline, total_at_yd}

Pool: full training-data set for 5-reg/6-reg, full 9.35M valid-K HDF5
for 7-reg (matching 04b).
"""
from __future__ import annotations

import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loaders as L
from simulator import simulate_perms_batch

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "processed"

# Tunable MC parameters. Picked for ~1-hour total wall-time at ~250 sims/sec.
N_MAIN_SAMPLES   = 60    # permutation orderings for main Shapley
N_MAIN_MARGINAL  = 192   # A15 — bumped from 64 (4× more than 04c)
N_PAIR_SAMPLES   = 100   # subset draws for Shapley-Taylor pairs
N_PAIR_MARGINAL  = 48    # valid completions per coalition for pair v(·)


def load_completion_pool(name: str) -> np.ndarray:
    """(M, l) int64 valid permutations for the on-manifold completion."""
    if name == "0x6D":
        ex = L.EXEMPLARS[name]
        h5p = ex.dir / "sampled_valid_permutations_7_gates_10000000_samples.h5"
        df = pd.read_hdf(h5p, key="df").drop_duplicates().reset_index(drop=True)
        return df.values.astype(np.int64)
    td = L.load_training_data(name)
    return np.stack([np.asarray(p, dtype=np.int64) for p in td["permutation"]])


def v_of_S(name: str, perm: np.ndarray, pool: np.ndarray,
           S_mask: np.ndarray, n_marginal: int, rng) -> tuple[float, float]:
    """E_completion[ simulator(x_S=yd, x_{-S} ~ pool) ] — returns (c, g) jointly."""
    idx = rng.integers(0, len(pool), size=n_marginal)
    out = pool[idx].copy()
    out[:, S_mask] = perm[S_mask]
    df = simulate_perms_batch(name, out)
    return float(df["circuit_score"].mean()), float(df["growth_score"].mean())


def main_shapley_sim(name: str, perm: np.ndarray, pool: np.ndarray,
                     *, n_samples: int, n_marginal: int,
                     rng=None) -> tuple[np.ndarray, np.ndarray, float, float]:
    if rng is None:
        rng = np.random.default_rng(42)
    l = len(perm)
    # v(∅) once with a larger sample (variance reduction)
    v_empty_c, v_empty_g = v_of_S(name, perm, pool,
                                  np.zeros(l, dtype=bool),
                                  n=512 if False else 1024, rng=rng) if False else \
                            v_of_S(name, perm, pool,
                                   np.zeros(l, dtype=bool),
                                   n_marginal=1024, rng=rng)
    shap_c = np.zeros(l, dtype=np.float64)
    shap_g = np.zeros(l, dtype=np.float64)
    total_sims = n_samples * l * n_marginal
    t0 = time.time(); done = 0
    for it in range(n_samples):
        order = rng.permutation(l)
        S_mask = np.zeros(l, dtype=bool)
        v_prev_c, v_prev_g = v_empty_c, v_empty_g
        for p in order:
            S_mask[p] = True
            vc, vg = v_of_S(name, perm, pool, S_mask, n_marginal, rng)
            shap_c[p] += vc - v_prev_c
            shap_g[p] += vg - v_prev_g
            v_prev_c, v_prev_g = vc, vg
            done += n_marginal
        if (it + 1) % 10 == 0:
            rate = done / (time.time() - t0 + 1e-9)
            eta = (total_sims - done) / max(rate, 1)
            print(f"  main: ordering {it + 1}/{n_samples}  "
                  f"({done:,}/{total_sims:,} sims, {rate:.0f}/s, eta {eta:.0f}s)")
    return shap_c / n_samples, shap_g / n_samples, v_empty_c, v_empty_g


def shapley_taylor_pair_sim(name: str, perm: np.ndarray, pool: np.ndarray,
                            i: int, j: int, *,
                            n_samples: int, n_marginal: int,
                            rng=None) -> tuple[float, float]:
    """Shapley-Taylor 2-body Φᵢⱼ on (circuit, growth) jointly."""
    if rng is None:
        rng = np.random.default_rng(1000 + i * 100 + j)
    l = len(perm)
    others = [k for k in range(l) if k not in (i, j)]
    acc_c = 0.0
    acc_g = 0.0
    for _ in range(n_samples):
        s_size = rng.integers(0, len(others) + 1)
        sel = (rng.choice(others, size=s_size, replace=False)
               if s_size > 0 else np.array([], dtype=int))
        m_base = np.zeros(l, dtype=bool)
        m_base[sel] = True
        mask_S = m_base.copy()
        mask_Si = m_base.copy(); mask_Si[i] = True
        mask_Sj = m_base.copy(); mask_Sj[j] = True
        mask_Sij = m_base.copy(); mask_Sij[i] = True; mask_Sij[j] = True
        # Use shared marginal draws for variance reduction
        idx = rng.integers(0, len(pool), size=n_marginal)
        drawn = pool[idx]

        def vboth(mask):
            out = drawn.copy()
            out[:, mask] = perm[mask]
            df = simulate_perms_batch(name, out)
            return (float(df["circuit_score"].mean()),
                    float(df["growth_score"].mean()))
        c_S,    g_S    = vboth(mask_S)
        c_Si,   g_Si   = vboth(mask_Si)
        c_Sj,   g_Sj   = vboth(mask_Sj)
        c_Sij,  g_Sij  = vboth(mask_Sij)
        acc_c += c_Sij - c_Si - c_Sj + c_S
        acc_g += g_Sij - g_Si - g_Sj + g_S
    return acc_c / n_samples, acc_g / n_samples


def analyse_one(name: str) -> dict:
    print(f"\n===== {name} — sim-valued Shapley-Taylor =====")
    pool = load_completion_pool(name)
    yd = L.get_yellow_dot_info(name)
    perm = np.asarray(yd["perm"], dtype=np.int64)
    l = len(perm)
    print(f"  pool size: {len(pool):,}  ·  l = {l}")

    # ---- main Shapley with n_marginal=256 (A15) ----
    print(f"  main Shapley (n_samples={N_MAIN_SAMPLES}, "
          f"n_marginal={N_MAIN_MARGINAL}) ...")
    rng_main = np.random.default_rng(42)
    main_c, main_g, v0_c, v0_g = main_shapley_sim(
        name, perm, pool,
        n_samples=N_MAIN_SAMPLES,
        n_marginal=N_MAIN_MARGINAL,
        rng=rng_main,
    )

    # ---- Shapley-Taylor pair Shapley (A1) ----
    n_pairs = l * (l - 1) // 2
    print(f"  Shapley-Taylor pairs ({n_pairs} pairs, "
          f"n_samples={N_PAIR_SAMPLES}, n_marginal={N_PAIR_MARGINAL}) ...")
    pair_c = np.zeros((l, l), dtype=np.float64)
    pair_g = np.zeros((l, l), dtype=np.float64)
    pair_t0 = time.time()
    for k_idx, (i, j) in enumerate(combinations(range(l), 2)):
        t_pair = time.time()
        pcv, pgv = shapley_taylor_pair_sim(
            name, perm, pool, i, j,
            n_samples=N_PAIR_SAMPLES, n_marginal=N_PAIR_MARGINAL,
        )
        pair_c[i, j] = pair_c[j, i] = pcv
        pair_g[i, j] = pair_g[j, i] = pgv
        print(f"    pair ({i},{j})   "
              f"Φc={pcv:+.3f}  Φg={pgv:+.4f}   "
              f"({time.time() - t_pair:.0f}s)   "
              f"[{k_idx + 1}/{n_pairs}]")
    print(f"  pairs done in {(time.time() - pair_t0)/60:.1f} min")

    # ---- yd ground-truth scores for the title block ----
    yd_row = simulate_perms_batch(name, [list(perm)])
    total_c = float(yd_row["circuit_score"].iloc[0])
    total_g = float(yd_row["growth_score"].iloc[0])

    slots = L.regulator_slot_metadata(name)
    out = {
        "name": name,
        "perm": perm.tolist(),
        "pool_size": int(len(pool)),
        "pool_source": ("full_valid_K_hdf5" if name == "0x6D"
                        else "full_training_data"),
        "slot_labels": [f"{r.repressor}-{r.rbs}"
                        for _, r in slots.iterrows()],
        "n_main_samples": N_MAIN_SAMPLES,
        "n_main_marginal": N_MAIN_MARGINAL,
        "n_pair_samples": N_PAIR_SAMPLES,
        "n_pair_marginal": N_PAIR_MARGINAL,
        "circuit": {
            "main_shap": main_c.tolist(),
            "pair_shap": pair_c.tolist(),
            "baseline": v0_c,
            "total_at_yd": total_c,
        },
        "growth": {
            "main_shap": main_g.tolist(),
            "pair_shap": pair_g.tolist(),
            "baseline": v0_g,
            "total_at_yd": total_g,
        },
    }
    out_path = PROC / f"shapley_taylor_sim_{name}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {out_path}")
    return out


def main():
    for name in L.EXEMPLARS:
        analyse_one(name)


if __name__ == "__main__":
    main()
