"""Focused S11 data, selection, normalization and rendering checks.

Run with the scientific Python environment. Optional --report PATH writes the
audit summary; --before-hashes PATH verifies the four attribution inputs were
not modified during a source-panel rebuild. No model/simulator rerun is needed.
"""
from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


with patch.dict(os.environ, {"DEEPCIRC_SCRATCH": tempfile.gettempdir()}):
    GENERATOR = load_module("s11_generator", ROOT / "topology/scripts/29_panel_c_shapley.py")
PANEL_A = load_module("s11_panel_a", HERE / "build_panel_d_stacked.py")
PANEL_B = load_module("s11_panel_b", HERE / "build_panel_e.py")
DESIGNS = json.loads(PANEL_A.JSON_PATH.read_text())["designs"]
PAIR_PATHS = [PANEL_B.MAIN_PROC / f"shapley_taylor_sim_{name}.json"
              for name in PANEL_B.TOPOLOGIES]
INPUT_PATHS = [PANEL_A.JSON_PATH, *PAIR_PATHS]
PARTS = ["AmeR/F1", "AmtR/A1", "BetI/E1", "BM3R1/B1", "BM3R1/B2", "BM3R1/B3",
         "HlyIIR/H1", "IcaRA/I1", "LitR/L1", "LmrA/N1", "PhlF/P1", "PhlF/P2",
         "PhlF/P3", "PsrA/R1", "QacR/Q1", "QacR/Q2", "SrpR/S1", "SrpR/S2",
         "SrpR/S3", "SrpR/S4"]


def hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in INPUT_PATHS}


def audit_summary():
    pairs = {}
    saturated = []
    for path in PAIR_PATHS:
        data = json.loads(path.read_text())
        name = data["name"]
        PANEL_B._load_pairwise(name)
        pairs[name] = {}
        for objective in PANEL_B.OBJECTIVES:
            row = data[objective]
            main = np.asarray(row["main_shap"])
            matrix = np.asarray(row["pair_shap"])
            upper = matrix[np.triu_indices(len(main), 1)]
            pairs[name][objective] = {
                "pairs": len(upper), "positive": int((upper > 0).sum()),
                "negative": int((upper < 0).sum()), "zero": int((upper == 0).sum()),
                "main_efficiency_residual": float(main.sum() - (row["total_at_yd"] - row["baseline"])),
                "pair_min": float(upper.min()), "pair_max": float(upper.max()),
            }
            if objective == "growth":
                saturated.extend({"target": name, "slot": i, "part": data["slot_labels"][i],
                                  "raw_value": float(value)}
                                 for i, value in enumerate(main) if abs(value) > 0.05)
    denominator_errors = [abs(sum(abs(v) for v in PANEL_A._fractional_phi(d, objective).values()) - 1)
                          for d in DESIGNS for objective in PANEL_B.OBJECTIVES]
    artifacts = []
    for name in ("panel_d", "panel_e"):
        for suffix in ("pdf", "svg", "png"):
            folder = "raster" if suffix == "png" else "vector"
            path = HERE.parent / "panels" / folder / f"{name}.{suffix}"
            if path.is_file():
                artifacts.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return {
        "audit_date": "2026-10-30", "scope": "Current setFinalvF S11 source panels; no Illustrator/master rebuild",
        "attribution_inputs_sha256": hashes(),
        "attribution_recomputed": False,
        "source_panel_artifacts": artifacts,
        "source_fixes": ["Raw c/g headers and conventional Shapley stripe labels",
                         "Signed point-estimate legend with matching grey/blue swatches",
                         "Unestimated diagonal labels, no Taylor-decomposition claim",
                         "Sign-preserving compact labels and unrounded colours",
                         "Required simulator inputs with scientific validation, no MLP fallback",
                         "Pareto-knee group label and replacement-attribution terminology",
                         "Selection generator uses log2/exp2; saved cohort unchanged"],
        "panel_a": {
            "target_entries": len(DESIGNS), "part_positions": sum(d["gate_count"] for d in DESIGNS),
            "unique_assignment_identities": len({(d["topology_id"], d["perm_idx"], tuple(d["gate_assignments"])) for d in DESIGNS}),
            "maximum_absolute_fraction_sum_error": max(denominator_errors),
            "cohort_matches_corrected_log2_selection": True,
            "all_30_entries_match_table_parts_sizes_and_scores": True,
            "circuit_positive_slots": sum(v > 0 for d in DESIGNS for v in d["phi_circuit"]),
            "max_circuit_entries_with_negative_growth": sum(any(v < 0 for v in d["phi_growth"]) for d in DESIGNS if d["group"] == "max-circ"),
            "knee_entries_with_negative_growth": sum(any(v < 0 for v in d["phi_growth"]) for d in DESIGNS if d["group"] == "knee"),
        },
        "panel_b": {"units": {"circuit": "raw c", "growth": "raw g"}, "checks": pairs,
                    "growth_colour_limits": [-0.05, 0.05], "growth_stripe_saturated_values": saturated,
                    "tiny_negative_growth_value": -0.000244782755214803,
                    "tiny_negative_display": PANEL_B._format_dp(-0.000244782755214803, 3),
                    "colour_values_rounded": False,
                    "uncertainty": "Monte Carlo point estimates; no confidence intervals stored",
                    "legacy_mlp_fallback": False},
    }


class ScientificContract(unittest.TestCase):
    def test_saved_arrays_and_bar_normalization(self):
        self.assertEqual(len(DESIGNS), 30)
        for d in DESIGNS:
            n = d["gate_count"]
            self.assertEqual([PARTS[i] for i in d["gate_assignments"]], d["part_names"])
            self.assertEqual(len({p.split("/")[0] for p in d["part_names"]}), n)
            for objective in PANEL_B.OBJECTIVES:
                values = np.asarray(d["phi_" + objective])
                self.assertEqual(values.shape, (n,))
                self.assertTrue(np.isfinite(values).all())
                self.assertGreater(np.abs(values).sum(), 0)
                fractions = PANEL_A._fractional_phi(d, objective)
                self.assertAlmostEqual(sum(abs(v) for v in fractions.values()), 1)
        zero = copy.deepcopy(DESIGNS[0])
        zero["phi_circuit"] = [0] * zero["gate_count"]
        self.assertTrue(all(v == 0 for v in PANEL_A._fractional_phi(zero, "circuit").values()))

    def test_selection_still_matches_saved_cohort_in_log2_units(self):
        self.assertEqual(GENERATOR.CIRC_LOG_FLOOR, 1.0)
        for group, selected in (("max-circ", GENERATOR._top15_maxcirc()),
                                ("knee", GENERATOR._top15_knee())):
            expected = {d["target"]: d for d in DESIGNS if d["group"] == group}
            self.assertEqual(set(selected.target), set(expected))
            for _, row in selected.iterrows():
                d = expected[row.target]
                self.assertEqual(row.topology_id, d["topology_id"])
                self.assertEqual(int(row.perm_idx), d["perm_idx"])
                self.assertEqual(ast.literal_eval(row.gate_assignments), d["gate_assignments"])
                self.assertAlmostEqual(row.circuit_score, d["mlp_circuit_score"], delta=0.003)

    def test_tables_match_saved_entries(self):
        for number, group in ((1, "max-circ"), (2, "knee")):
            path = HERE.parents[1] / f"tabS0{number}/final/table_S{number}.csv"
            # S2 has a separate frequency table after the first blank line.
            rows = list(csv.DictReader(path.read_text().split("\n\n")[0].splitlines()))[:15]
            expected = {d["target"]: d for d in DESIGNS if d["group"] == group}
            self.assertEqual(len(rows), 15)
            for row in rows:
                d = expected[row["target"]]
                self.assertCountEqual(row["part_names"].split(" → "), d["part_names"])
                self.assertEqual(row["size"], f"{d['gate_count']}-reg")
                self.assertAlmostEqual(float(row["circuit_score"]), d["mlp_circuit_score"], delta=0.0051)
                self.assertAlmostEqual(float(row["growth_score"]), d["mlp_growth_score"], delta=0.00051)

    def test_replacement_mean_includes_original_and_respects_families(self):
        # Independent hand-check: PhlF/P2, SrpR/S4, QacR/Q2, BetI/E1.
        gates = [11, 19, 15, 2]
        expected_sets = [[0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                         [0, 1, 3, 4, 5, 6, 7, 8, 9, 13, 16, 17, 18, 19],
                         [0, 1, 3, 4, 5, 6, 7, 8, 9, 13, 14, 15],
                         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 13]]
        families = [p.split("/")[0] for p in PARTS]
        class FamilyCheckingScore(torch.nn.Module):
            def forward(self, x):
                indices = x.reshape(-1, 4, 20).argmax(dim=2)
                for row in indices.tolist():
                    if len({families[i] for i in row}) != 4:
                        raise AssertionError("family-invalid replacement reached scorer")
                return (indices.double() ** 2).sum(dim=1, keepdim=True)
        base, values = GENERATOR.slot_shapleys(FamilyCheckingScore(), gates, 4)
        self.assertEqual(base, sum(g * g for g in gates))
        for i, allowed in enumerate(expected_sets):
            self.assertIn(gates[i], allowed)
            self.assertAlmostEqual(values[i], gates[i] ** 2 - np.mean(np.square(allowed)))

    def test_pair_arrays_and_shapley_efficiency(self):
        for name in PANEL_B.TOPOLOGIES:
            data = PANEL_B._load_pairwise(name)
            for objective in PANEL_B.OBJECTIVES:
                pair = data["pairwise_" + objective]
                self.assertTrue(np.array_equal(pair, pair.T))
                self.assertTrue(np.all(np.diag(pair) == 0))
                self.assertTrue(np.isfinite(pair).all())

    def test_missing_simulator_data_never_uses_legacy_mlp(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(PANEL_B, "MAIN_PROC", Path(temp)):
            (Path(temp) / "pairwise_0x2B.json").write_text("{}")
            with self.assertRaisesRegex(FileNotFoundError, "Legacy MLP"):
                PANEL_B._load_pairwise("0x2B")

    def test_invalid_pair_payloads_are_rejected(self):
        source = json.loads(PAIR_PATHS[0].read_text())
        mutations = [lambda d: d["circuit"]["pair_shap"][0].__setitem__(1, float("nan")),
                     lambda d: d["growth"]["pair_shap"][0].__setitem__(1, 0.9),
                     lambda d: d["circuit"]["pair_shap"][0].__setitem__(0, 1),
                     lambda d: d["growth"]["main_shap"].pop(),
                     lambda d: d["growth"]["main_shap"].__setitem__(0, 9)]
        with tempfile.TemporaryDirectory() as temp, patch.object(PANEL_B, "MAIN_PROC", Path(temp)):
            for mutation in mutations:
                data = copy.deepcopy(source)
                mutation(data)
                (Path(temp) / "shapley_taylor_sim_0x2B.json").write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    PANEL_B._load_pairwise("0x2B")

    def test_small_values_keep_sign_and_render_raw_colour(self):
        self.assertEqual(PANEL_B._format_dp(-0.000244782755214803, 3), "-2e-4")
        self.assertEqual(PANEL_B._format_dp(0.000244782755214803, 3), "+2e-4")
        self.assertEqual(PANEL_B._format_dp(-1e-8, 3), "-1e-8")
        self.assertEqual(PANEL_B._format_dp(0, 3), "0")
        self.assertEqual(PANEL_B._format_dp(0.0123, 3), "+0.012")
        fig, axes = plt.subplots(1, 2)
        try:
            data = PANEL_B._load_pairwise("0x6D")
            PANEL_B._panel_heatmap(*axes, name="0x6D", obj="growth", data=data)
            self.assertIn("-2e-4", [t.get_text() for t in axes[0].texts])
            self.assertNotIn("log", axes[0].get_title(loc="left"))
            self.assertIn("Shapley", axes[1].get_title())
            value = data["pairwise_growth"][1, 3]
            norm = PANEL_B.TwoSlopeNorm(vmin=-0.05, vcenter=0, vmax=0.05)
            self.assertEqual(axes[0].patches[1 * 7 + 3].get_facecolor(), PANEL_B.DIVERGING_CMAP(norm(value)))
        finally:
            plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    parser.add_argument("--before-hashes", type=Path)
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ScientificContract))
    if not result.wasSuccessful():
        raise SystemExit(1)
    summary = audit_summary()
    if args.before_hashes:
        assert hashes() == json.loads(args.before_hashes.read_text()), "Attribution input changed"
        summary["attribution_inputs_unchanged_since_before_rebuild"] = True
    summary["validation"] = {"tests_run": result.testsRun, "failures": 0, "errors": 0}
    if args.report:
        args.report.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Audit report: {args.report}")
