#!/usr/bin/env python3
"""Smoke-test the loaders module.

Import path: this script adds its own directory to sys.path so it can
`import loaders`. Other scripts in this folder use the same pattern.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loaders as L


def main():
    for name in L.EXEMPLARS:
        ex = L.EXEMPLARS[name]
        print(f"\n=== {name} (folder={ex.folder}, suffix={ex.suffix}, n_regs={ex.n_regs}) ===")
        yd = L.get_yellow_dot_info(name)
        print(f"  yellow dot: perm={yd['perm']}, circuit={yd['circuit_score']:.4f}, "
              f"growth={yd['growth_score']:.4f}, interference={yd['interference']}")
        slots = L.regulator_slot_metadata(name)
        print(f"  regulator slots ({len(slots)}):")
        for _, r in slots.iterrows():
            print(f"    slot {r.slot_idx} (node {r.node_id}): {r.repressor}-{r.rbs}  "
                  f"(ymaxa={r.ymaxa:.3f}, Ka={r.Ka:.3f}, n={r.n_hill:.3f})")

        td = L.load_training_data(name)
        shp = L.load_scored_high_predicted(name)
        pph = L.load_predicted_high_perms(name)
        top = L.load_top_ranked(name)
        print(f"  training_data: {len(td):,} rows, "
              f"c mean={td.circuit_score.mean():.3f}, g mean={td.growth_score.mean():.3f}, "
              f"interference frac={td.interference_flag.mean():.3f}")
        print(f"  scored_high_predicted: {len(shp):,} rows, "
              f"c mean={shp.circuit_score.mean():.3f}")
        print(f"  predicted_high_perms npy: shape={pph.shape}")
        print(f"  top-ranked: {len(top)} rows, top c={top.circuit_score.iloc[0]:.3f}")

        # Model inference check
        cm = L.load_circuit_mlp(name)
        gm = L.load_growth_mlp(name)
        perm = yd["perm"]
        c_hat = L.predict(cm, [perm])[0]
        g_hat = L.predict(gm, [perm])[0]
        nn_saved = L.load_nn_prediction_of_best(name)
        print(f"  MLP check: perm={perm}")
        print(f"    predicted now: c={c_hat:.4f}, g={g_hat:.4f}")
        print(f"    predicted saved: c={nn_saved['circuit_pred']:.4f}, g={nn_saved['growth_pred']:.4f}")
        assert abs(c_hat - nn_saved["circuit_pred"]) < 1e-3, "circuit MLP mismatch"
        assert abs(g_hat - nn_saved["growth_pred"]) < 1e-3, "growth MLP mismatch"
        print("    OK (matches saved predictions to <1e-3)")


if __name__ == "__main__":
    main()
