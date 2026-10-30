"""Scientific contract tests for circuit-presence enrichment.

Run with: python -m unittest discover -s deepcirc_topology/tests -v
"""
from __future__ import annotations

from collections import defaultdict
import importlib.util
import contextlib
import io
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from _presence_enrichment import presence_statistics  # noqa: E402

SPEC = importlib.util.spec_from_file_location("l2_graph_role", SCRIPTS / "09_l2_graph_role.py")
L2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(L2)


def fresh_counters():
    return (
        {task: defaultdict(L2.cell_factory) for task in ("circuit_log", "toxicity")},
        {task: defaultdict(lambda: defaultdict(L2.cell_factory_target))
         for task in ("circuit_log", "toxicity")},
    )


class PresenceStatisticsTests(unittest.TestCase):
    def test_unequal_tier_sizes_with_equal_presence_probabilities(self):
        result = presence_statistics(5, 10, 10, 20)
        self.assertEqual(float(result["top_probability"]), 0.5)
        self.assertEqual(float(result["bottom_probability"]), 0.5)
        self.assertEqual(float(result["log2_presence_enrichment"]), 0.0)
        self.assertNotEqual(np.log2(6 / 11), 0)

    def test_equal_exposure_recovers_legacy_smoothed_count_ratio(self):
        result = presence_statistics([0, 5, 50], [10, 5, 2], 100, 100)
        np.testing.assert_allclose(result["log2_presence_enrichment"],
                                   np.log2(np.array([1, 6, 51]) / [11, 6, 3]))

    def test_real_amer_counterexample_changes_sign(self):
        result = presence_statistics(477162, 340642, 7700606, 4374044)
        self.assertGreater(np.log2(477163 / 340643), 0)
        self.assertLess(float(result["log2_presence_enrichment"]), 0)
        expected = np.log2((477163 / 7700608) / (340643 / 4374046))
        self.assertAlmostEqual(float(result["log2_presence_enrichment"]), expected)

    def test_invalid_counts_and_missing_exposure_fail(self):
        bad = [(-1, 1, 10, 10), (0.5, 1, 10, 10), (np.nan, 1, 10, 10),
               (np.inf, 1, 10, 10), (11, 1, 10, 10), (0, 0, 0, 10),
               (True, 1, 10, 10), ("1", 1, 10, 10), (1, 1, 2**54, 10)]
        for args in bad:
            with self.subTest(args=args), self.assertRaises(ValueError):
                presence_statistics(*args)
        with self.assertRaisesRegex(ValueError, "broadcastable"):
            presence_statistics([1, 2], [1, 2, 3], 10, 10)


class PresenceAggregationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "predictions.npz"
        # The repeated part in row 2 deliberately checks the counting helper's
        # row-wise deduplication; production calls validate unique TF families.
        self.assignments = np.array([[0, 1, 2], [1, 0, 3], [1, 1, 4], [2, 3, 4]], dtype=np.int8)
        np.savez(self.path, gate_assignments=self.assignments,
                 circuit_pred=np.array([0.0, 0.0, 1.0, 2.0]),
                 toxicity_pred=np.array([1.0, 1.0, 0.0, 0.0]))
        context = {"depth_to_output": 2, "depth_from_input": 1,
                   "in_degree": 1, "out_degree": 1, "node_type": "NOT"}
        other = {**context, "depth_to_output": 1}
        self.fingerprints = [context, context.copy(), other]
        self.key = L2.fingerprint_key(context)
        self.meta = {"topology_id": "fixture", "source": ["0x2B", "0x4D"]}
        self.counters, self.by_target = fresh_counters()

    def tearDown(self):
        self.temp.cleanup()

    def aggregate(self, **kwargs):
        return L2.aggregate_one_topology(self.path, self.meta, self.fingerprints,
                                         self.counters, self.by_target,
                                         top_quantile=0.95, bot_quantile=0.05, **kwargs)

    def test_repeated_context_counts_circuits_once_and_keeps_both_task_masks(self):
        summary = self.aggregate()
        circuit = self.counters["circuit_log"][self.key]
        growth = self.counters["toxicity"][self.key]
        self.assertEqual(circuit["top_design_count"], 1)
        self.assertEqual(circuit["bottom_design_count"], 2)
        self.assertEqual(int(circuit["top"].sum()), 2)  # Legacy position count remains.
        self.assertEqual(int(circuit["bot"].sum()), 4)
        self.assertEqual(circuit["top_part_design_count"][2], 1)
        self.assertEqual(circuit["top_part_design_count"][3], 1)
        self.assertEqual(growth["top_design_count"], 2)
        self.assertEqual(growth["bottom_design_count"], 2)
        self.assertEqual(growth["bottom_part_design_count"][1], 1)  # Repeated part counted once.
        self.assertEqual(growth["bot"][1], 2)  # Legacy occurrence semantics retained.
        self.assertEqual(summary["fingerprint_multiplicity"][self.key], 2)
        self.assertEqual(summary["tier_statistics"]["circuit_log"]["bottom_threshold_ties"], 2)
        self.assertEqual(summary["tier_statistics"]["toxicity"]["top_threshold_ties"], 2)
        self.assertEqual(summary["n_top_designs"], 1)  # Legacy circuit-only field retained.

    def test_multiple_targets_do_not_duplicate_aggregate_exposure(self):
        self.aggregate()
        self.assertEqual(self.counters["circuit_log"][self.key]["top_design_count"], 1)
        for source in self.meta["source"]:
            self.assertEqual(self.by_target["circuit_log"][source][self.key]["top_design_count"], 1)
        names = [f"family{i}/R{i}" for i in range(20)]
        aggregate = L2.emit_presence_dataframe(self.counters, names)
        targeted = L2.emit_presence_dataframe(self.by_target, names, by_target=True)
        self.assertEqual(len(targeted), 2 * len(aggregate))
        self.assertEqual(set(aggregate["n_targets"]), {2})
        self.assertEqual(set(targeted["n_targets"]), {1})
        self.assertNotIn("log_odds", aggregate.columns)
        self.assertNotIn("total_designs", aggregate.columns)
        legacy = L2.emit_aggregate_dataframe(self.counters, names)
        np.testing.assert_allclose(legacy["log_odds"],
                                   np.log2((legacy["top_count"] + 1) / (legacy["bot_count"] + 1)))

    def test_all_ties_are_retained_and_overlap_is_recorded(self):
        np.savez(self.path, gate_assignments=self.assignments,
                 circuit_pred=np.ones(4), toxicity_pred=np.ones(4))
        summary = self.aggregate()
        for task in ("circuit_log", "toxicity"):
            info = summary["tier_statistics"][task]
            self.assertEqual(info["top_design_count"], 4)
            self.assertEqual(info["bottom_design_count"], 4)
            self.assertEqual(info["overlap_design_count"], 4)

    def test_family_conflict_fails_before_counters_are_mutated(self):
        with self.assertRaisesRegex(ValueError, "same regulator family"):
            self.aggregate(part_families=[f"family{i}" for i in range(20)])
        self.assertFalse(self.counters["circuit_log"])
        self.assertFalse(self.counters["toxicity"])

    def test_out_of_library_part_fails_before_counting(self):
        invalid = self.assignments.copy()
        invalid[0, 0] = 20
        np.savez(self.path, gate_assignments=invalid,
                 circuit_pred=np.arange(4), toxicity_pred=np.arange(4))
        with self.assertRaisesRegex(ValueError, "out-of-library"):
            self.aggregate()

    def cli_fixture(self):
        root = Path(self.temp.name)
        context = self.fingerprints[0]
        fingerprints = [context, context.copy(), {**context, "depth_to_output": 3},
                        {**context, "depth_to_output": 4}]
        np.savez(root / "fixture_0.npz",
                 gate_assignments=np.array([[0, 1, 2, 3], [1, 0, 3, 4],
                                            [2, 3, 4, 5], [3, 4, 5, 6]], dtype=np.int8),
                 circuit_pred=np.array([0.0, 0.0, 1.0, 2.0]),
                 toxicity_pred=np.array([1.0, 1.0, 0.0, 0.0]))
        with (root / "population.pkl").open("wb") as f:
            pickle.dump({"topologies": [{**self.meta, "regulator_count": 4,
                                          "fingerprints": fingerprints}]}, f)
        with (root / "qc.pkl").open("wb") as f:
            pickle.dump({"records": [{"topology_id": "fixture", "tier": "A1",
                                      "best_perm_idx": 0}]}, f)
        (root / "parts.json").write_text(json.dumps({"Repressor": [f"family{i}" for i in range(20)],
                                                     "RBS": [f"R{i}" for i in range(20)]}))
        args = ["09_l2_graph_role.py", "--population_manifest", str(root / "population.pkl"),
                "--qc_tiers", str(root / "qc.pkl"), "--predictions_root", str(root),
                "--part_response_data", str(root / "parts.json"),
                "--output_root", str(root / "output"), "--top_quantile", "0.95",
                "--bot_quantile", "0.05"]
        return root, args

    def test_cli_emits_versioned_outputs_without_relabeling_legacy_values(self):
        root, args = self.cli_fixture()
        with patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
            L2.main()
        out = root / "output"
        self.assertEqual(len(list(out.iterdir())), 6)
        summary = json.loads((out / "l2_presence_summary_v2.json").read_text())
        self.assertEqual(summary["schema"], "deepcirc.part_presence_enrichment.v2")
        self.assertTrue(summary["family_unique_assignments_validated"])
        self.assertEqual(set(summary["per_topology"][0]["tier_statistics"]), {"circuit_log", "toxicity"})
        legacy = L2.pd.read_csv(out / "l2_enrichment.csv")
        np.testing.assert_allclose(legacy["log_odds"],
                                   np.log2((legacy["top_count"] + 1) / (legacy["bot_count"] + 1)))
        presence = L2.pd.read_csv(out / "l2_presence_enrichment_v2.csv")
        self.assertIn("log2_presence_enrichment", presence)
        self.assertNotIn("log_odds", presence)

    def test_missing_eligible_predictions_cannot_emit_partial_cohort(self):
        root, args = self.cli_fixture()
        (root / "fixture_0.npz").unlink()
        with patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(SystemExit, "incomplete enrichment cohort"):
                L2.main()
        self.assertEqual(list((root / "output").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
