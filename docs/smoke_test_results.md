# Smoke test results (Phase 5)

Verified end-to-end figure reproducibility from a fresh `git clone` of the
repository plus the in-repo Tier 0 data.

## Environment

- Conda env created from `environment.yml`: `deepcirc-interp` on Python 3.12.13
- Package installed via `pip install -e .`
- Submodule `upstream/DeepCirc` initialized at pinned commit `ad5b1494`
- No external downloads (no Zenodo fetch); Tier 0 in-repo data only

## Canonical panel scripts (referenced by manifests)

| Figure | Panel script | Result | Notes |
|---|---|---|---|
| figS10 | `build_panel_a.py` | PASS | procedural schematic, no data |
| figS10 | `build_panel_b.py` | PASS | reads `size_tier_designs/dense_percentile_sweep.json` |
| figS10 | `build_panel_c.py` | PASS | reads `size_tier_designs/dense_percentile_sweep.json` |
| figS10 | `build_panel_d.py` | PASS | reads `motif_tier_analysis/structural_iso_by_size_*.csv` |
| figS11 | `build_panel_a.py` | PASS | reads `motif_tier_analysis/typed_expansions_by_size_*.csv` |
| figS12 | `build_panel_b.py` | PASS | 3×3 NPN grid, reads `pareto/all_topology_fronts.csv` |
| figS13 | `build_panel_c.py` | PASS | reads `pareto/knee_designs.csv` + `topology_graphs.json` + `panel_c_shapley/shapley_per_design.json` |
| figS14 | `build_panel_{a,b,c}.py` | PASS | reads `l2_top05/l2_enrichment.csv` (both v1 + v2 palettes pass) |
| figS15 | `build_panel_d.py` | PASS | reads `panel_c_shapley/shapley_per_design.json` |
| figS15 | `build_panel_e.py` | PASS | reads `interp_processed/shapley_taylor_sim_0x{2B,17,6D}.json` |

**Canonical panel pass rate: 14 / 14.**

## Manifest composition

`figures/scripts/build_figure.py --manifest figures/setFinal/figSXX/manifest.yaml`
produces a single composed PDF + PNG + linked-asset JSON per figure:

| Figure | PDF size | PNG size |
|---|---|---|
| figure_S10 | 2.5 MB | 960 KB |
| figure_S11 | 2.4 MB | – |
| figure_S12 | 7.0 MB | – |
| figure_S13 | 5.1 MB | – |
| figure_S14 | 8.6 MB | – |
| figure_S15 | 9.7 MB | – |

**Manifest pass rate: 6 / 6.**

## Non-canonical alternate-render variants (not in any manifest)

The Phase 3 migration brought over 8 alternate-render variants that were
sketches / drafts / archival comparisons during figure development. None are
referenced by `manifest.yaml`. Most either depend on Tier 1 / Tier 2 data not
shipped in Tier 0 or contain a pre-existing color-dict KeyError. They do not
affect the published figure assembly.

| Figure | Variant | Status | Reason |
|---|---|---|---|
| figS12 | `build_panel_b_per_npn.py` | FAIL | requires Tier 1 `pareto/` data (alternate render) |
| figS13 | `build_panel_c_draft.py` | FAIL | Tier 1 dependency |
| figS13 | `build_panel_c_part_counts.py` | FAIL | Tier 1 dependency |
| figS13 | `build_panel_c_top5_highlight.py` | FAIL | Tier 1 dependency |
| figS13 | `build_panel_draft_c_knee_vs_maxcirc.py` | FAIL | Tier 1 dependency |
| figS15 | `build_panel_d_mock_heatmap.py` | FAIL | Tier 1 dependency |
| figS15 | `build_panel_d_mock_stacked.py` | FAIL | KeyError: `TOP5_COLORS["PsrA/R1"]` (real bug, pre-existing) |
| figS15 | `build_panel_dv0.py` | FAIL | Tier 2 exemplar pkl required (`optimal_topology_with_parts_assigned_0x2B_0_0.pkl`) |

These can be left in for archival reference or removed in a Phase 6 polish
pass. Recommendation: defer the decision to the user during Phase 6.

## Reproduction (this run)

```bash
git clone --recurse-submodules https://github.com/V-SCode/DeepCirc-interp.git
cd DeepCirc-interp
conda env create -f environment.yml
conda activate deepcirc-interp
pip install -e .

# Build all 6 figures from Tier 0 in-repo data:
make figures

# Verify outputs:
ls figures/setFinal/figS*/final/*.pdf
```

## Notes

- The fresh conda env was created from `environment.yml` exactly as published.
  No manual package additions were needed.
- `pip install -e .` resolved cleanly; the `pyproject.toml` package layout
  reaches every `from topology.X import ...`, `from interp.X import ...`,
  `from figures.X import ...` statement after Phase 3 import rewrites.
- The Tier 1 Pareto data (`knee_designs.csv`, `all_topology_fronts.csv`,
  `cross_target_portfolio.csv`, `pareto_summary.json`) was added to Tier 0
  during Phase 5 because canonical panels figS12/panel_b and figS13/panel_c
  depend on it; total addition was ~1 MB which is well within Tier 0 budget.
