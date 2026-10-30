"""Integration checks for current S10d/e plots and Table S3.

Plots are captured in memory; these tests never replace scientific artwork.
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
import numpy as np
import pandas as pd

VF_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from figures.setFinalvF._shared import presence_data


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, VF_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def measured_log_ratio(rows):
    top = (rows.top_part_design_count.to_numpy() + 1) / (
        rows.top_design_count.to_numpy() + 2)
    bottom = (rows.bottom_part_design_count.to_numpy() + 1) / (
        rows.bottom_design_count.to_numpy() + 2)
    return np.log2(top / bottom)


class PresencePanelIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = pd.read_csv(presence_data.DATA_PATH)

    def test_loader_preserves_complete_publication_context_set(self):
        contexts = []
        for task in ['toxicity', 'circuit_log']:
            rows = presence_data.load_presence_data(task)
            self.assertEqual(len(rows), 1300)
            self.assertEqual(rows.part_idx.nunique(), 20)
            self.assertEqual(rows.fp_key.nunique(), 65)
            self.assertTrue(rows.groupby('part_idx').fp_key.nunique().eq(65).all())
            contexts.append(set(rows.fp_key))
            np.testing.assert_allclose(rows.log2_presence_enrichment,
                                       measured_log_ratio(rows),
                                       atol=1e-12, rtol=1e-12)
        self.assertEqual(contexts[0], contexts[1])

    def assert_corruption_rejected(self, data, message):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'corrupted.csv'
            data.to_csv(source, index=False)
            with patch.object(presence_data, 'DATA_PATH', source):
                with self.assertRaisesRegex(ValueError, message):
                    presence_data.load_presence_data('circuit_log')

    def test_legacy_count_ratio_cannot_silently_replace_presence_metric(self):
        data = self.original.copy()
        data['log2_presence_enrichment'] = data.raw_log2_count_ratio_reference
        self.assert_corruption_rejected(data, 'normalized probability ratio')

    def test_changed_circuit_denominator_is_detected(self):
        data = self.original.copy()
        index = data.index[data.task == 'circuit_log'][0]
        data.loc[index, 'top_design_count'] *= 2
        self.assert_corruption_rejected(data, 'Probabilities do not match')

    def test_duplicate_context_part_is_rejected(self):
        duplicate = self.original[self.original.task == 'circuit_log'].iloc[[0]]
        self.assert_corruption_rejected(pd.concat([self.original, duplicate]),
                                        'Duplicate part/context')

    def test_missing_publication_part_is_rejected(self):
        data = self.original[~((self.original.task == 'circuit_log') &
                              (self.original.part_idx == 0))]
        self.assert_corruption_rejected(data, '65 eligible contexts')

    def test_missing_schema_field_is_rejected(self):
        self.assert_corruption_rejected(self.original.drop(columns='top_design_count'),
                                        'Missing presence-enrichment fields')

    def test_both_panels_plot_actual_normalized_values(self):
        for panel, task in [('d', 'toxicity'), ('e', 'circuit_log')]:
            with self.subTest(panel=panel):
                module = load_module('panel_' + panel,
                    f'figS10/scripts/build_panel_{panel}.py')
                captured = []
                def capture(figure, *_args, **_kwargs):
                    captured.append(figure)
                    return {}
                try:
                    with patch.object(module, 'save_panel', side_effect=capture):
                        module.build_panel()
                    self.assertEqual(len(captured), 1)
                    axes = captured[0].axes[0]
                    dots = [obj for obj in axes.collections
                            if isinstance(obj, PathCollection)]
                    labels = [label.get_text() for label in axes.get_yticklabels()]
                    self.assertEqual(len(dots), 20)
                    self.assertEqual(len(labels), 20)
                    rows = presence_data.load_presence_data(task)
                    for label, points in zip(labels, dots):
                        actual = np.asarray(points.get_offsets())[:, 0]
                        expected = measured_log_ratio(rows[rows.part_name == label])
                        self.assertEqual(len(actual), 65)
                        np.testing.assert_allclose(np.sort(actual), np.sort(expected),
                                                   atol=1e-12, rtol=1e-12)
                finally:
                    for figure in captured:
                        plt.close(figure)

    def test_table_and_saved_csv_match_means_of_actual_probability_ratios(self):
        module = load_module('table_s3', 'tabS03/scripts/build_table.py')
        built = module.build_table().set_index('part_name')
        saved = pd.read_csv(VF_ROOT / 'tabS03/final/table_S3.csv').set_index('part_name')
        rows = presence_data.load_presence_data('circuit_log')
        rows['checked_value'] = measured_log_ratio(rows)
        depth = rows.depth_to_output.to_numpy(dtype=int)
        rows['checked_bucket'] = rows.node_type + '-' + np.where(
            depth >= 3, 'early', np.where(depth == 2, 'middle', 'late'))
        expected = rows.groupby(['part_name', 'checked_bucket']).checked_value.mean().unstack()
        self.assertEqual(expected.shape, (20, 6))
        for actual in [built, saved]:
            aligned = expected.reindex(index=actual.index, columns=actual.columns).round(2)
            np.testing.assert_allclose(actual.to_numpy(), aligned.to_numpy(),
                                       atol=1e-12, rtol=0)
        self.assertEqual(saved.loc['AmeR/F1', 'NOR-middle'], -0.01)


if __name__ == '__main__':
    unittest.main()
