"""Scientific checks for archived-count recovery and circuit-presence migration."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np
from sympy import Matrix

SPEC = importlib.util.spec_from_file_location(
    'presence_builder', Path(__file__).with_name('build_presence_enrichment.py'))
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class ExactRecoveryTests(unittest.TestCase):
    def test_nonunique_topologies_but_unique_circuit_context_total(self):
        # Only x1+x2 is known; a context present in both topologies is identified.
        C = Matrix([[1, 1], [2, 2]])
        B = Matrix([[1, 1]])
        observed = Matrix([[10, 20], [20, 40]])
        result, _, pivots, free = builder.exact_context_totals(C, B, observed)
        self.assertEqual(result, Matrix([[10, 20]]))
        self.assertEqual(len(pivots), 1)
        self.assertEqual(len(free), 1)

    def test_unidentifiable_circuit_denominator_is_rejected(self):
        # Slot total x1+2*x2 cannot determine unique circuits x1+x2.
        with self.assertRaisesRegex(ValueError, 'not identifiable'):
            builder.exact_context_totals(
                Matrix([[1, 2]]), Matrix([[1, 1]]), Matrix([[10, 20]]))

    def test_inconsistent_source_counts_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Inconsistent'):
            builder.exact_context_totals(
                Matrix([[1], [1]]), Matrix([[1]]), Matrix([[10], [11]]))

    def test_fractional_circuit_totals_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'positive integers'):
            builder.exact_context_totals(
                Matrix([[2]]), Matrix([[1]]), Matrix([[5, 7]]))


class ArchivedCohortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame, cls.denominators, cls.proof = builder.build_presence_data(
            builder.DEFAULT_INPUT)

    def test_exact_proof_and_original_inputs(self):
        self.assertTrue(all(self.proof['exact_checks'].values()))
        self.assertEqual(self.proof['exact_rank'], 211)
        self.assertEqual(self.proof['nullity'], 4)
        self.assertEqual(len(self.denominators), 378)
        for name, digest in self.proof['input_sha256'].items():
            current = hashlib.sha256((builder.DEFAULT_INPUT / name).read_bytes()).hexdigest()
            self.assertEqual(current, digest)

    def test_original_eligible_contexts_preserved(self):
        eligible = self.frame[builder.eligibility(self.frame)]
        for _, rows in eligible.groupby('task'):
            self.assertEqual(len(rows), 1300)
            self.assertEqual(rows.fp_key.nunique(), 65)
            self.assertEqual(rows.part_idx.nunique(), 20)

    def test_actual_counterexample_uses_circuit_denominators(self):
        key = 'd2o=3|dfi=4|in=2|out=1|pred=0,3|succ=2|type=NOR'
        row = self.frame[(self.frame.task == 'circuit_log') &
                         (self.frame.fp_key == key) &
                         (self.frame.part_name == 'AmeR/F1')].iloc[0]
        self.assertEqual(row.top_design_count, 7700606)
        self.assertEqual(row.bottom_design_count, 4374044)
        self.assertGreater(row.raw_log2_count_ratio_reference, 0)
        self.assertLess(row.log2_presence_enrichment, 0)
        expected = np.log2(((477162 + 1) / (7700606 + 2)) /
                          ((340642 + 1) / (4374044 + 2)))
        self.assertAlmostEqual(row.log2_presence_enrichment, expected, places=13)

    def test_all_probabilities_are_binary_presence_rates(self):
        frame = self.frame
        self.assertTrue(frame.top_probability.between(0, 1).all())
        self.assertTrue(frame.bottom_probability.between(0, 1).all())
        np.testing.assert_allclose(
            frame.top_probability,
            (frame.top_part_design_count + 1) / (frame.top_design_count + 2),
            rtol=0, atol=0)
        np.testing.assert_allclose(
            frame.bottom_probability,
            (frame.bottom_part_design_count + 1) / (frame.bottom_design_count + 2),
            rtol=0, atol=0)
        self.assertNotIn('log_odds', frame.columns)
        self.assertNotIn('total_designs', frame.columns)

    def test_repeated_context_slots_count_each_circuit_once(self):
        summary = json.loads((builder.DEFAULT_INPUT / 'l2_summary.json').read_text())
        topology = next(t for t in summary['per_topology']
                        if len(set(t['fingerprint_keys'])) < len(t['fingerprint_keys']))
        context = next(k for k in topology['fingerprint_keys']
                       if topology['fingerprint_keys'].count(k) > 1)
        expected_top = sum(t['n_top_designs'] for t in summary['per_topology']
                           if context in t['fingerprint_keys'])
        slot_top = sum(t['n_top_designs'] * t['fingerprint_keys'].count(context)
                       for t in summary['per_topology'])
        row = self.denominators[(self.denominators.task == 'circuit_log') &
                                (self.denominators.fp_key == context)].iloc[0]
        self.assertEqual(row.top_circuits, expected_top)
        self.assertGreater(slot_top, row.top_circuits)

    def test_fresh_producer_emitter_matches_migration_contract(self):
        spec = importlib.util.spec_from_file_location(
            'presence_source_producer', builder.REPO_ROOT /
            'topology/scripts/09_l2_graph_role.py')
        producer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(producer)
        counters = defaultdict(dict)
        summary = json.loads((builder.DEFAULT_INPUT / 'l2_summary.json').read_text())
        part_names = self.frame[['part_idx', 'part_name']].drop_duplicates().sort_values(
            'part_idx').part_name.tolist()
        for (task, context), group in self.frame.groupby(['task', 'fp_key']):
            group = group.sort_values('part_idx')
            cell = producer.cell_factory()
            cell['top_part_design_count'] = group.top_part_design_count.to_numpy()
            cell['bottom_part_design_count'] = group.bottom_part_design_count.to_numpy()
            cell['top_design_count'] = int(group.top_design_count.iloc[0])
            cell['bottom_design_count'] = int(group.bottom_design_count.iloc[0])
            for topology in summary['per_topology']:
                if context in topology['fingerprint_keys']:
                    cell['topologies'].add(topology['topology_id'])
                    source = topology['source']
                    cell['targets'].update(source if isinstance(source, list) else [source])
            counters[task][context] = cell
        emitted = producer.emit_presence_dataframe(counters, part_names)
        require_columns = set(emitted.columns)
        self.assertTrue(require_columns.issubset(self.frame.columns))
        keys = ['task', 'fp_key', 'part_idx']
        left = emitted.sort_values(keys).reset_index(drop=True)
        right = self.frame.sort_values(keys).reset_index(drop=True)
        for column in ['top_design_count', 'bottom_design_count',
                       'top_part_design_count', 'bottom_part_design_count',
                       'part_tier_support', 'n_topologies', 'n_targets',
                       'top_probability', 'bottom_probability',
                       'log2_presence_enrichment']:
            np.testing.assert_array_equal(left[column], right[column])
        for key, value in producer.PRESENCE_METADATA.items():
            self.assertEqual(self.proof[key], value)


if __name__ == '__main__':
    unittest.main()
