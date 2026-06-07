# figures/

Vector-first, mm-precise, manifest-driven figure-assembly pipeline for the
DeepCirc paper supplementary figures **S10–S15**. Each figure is one folder
under `setFinal/`, with panel build scripts producing PDF + SVG + PNG, then a
YAML manifest composing them into the final figure.

## Folder layout (per figure)

```
setFinal/figSXX/
  scripts/         build_panel_<id>.py per panel
  panels/vector/   panel_<id>.pdf + .svg (canonical, vector)
  panels/raster/   panel_<id>.png (preview)
  final/           figure_SXX.pdf + .png + linked_assets.json
  manifest.yaml    mm-precise composition
  data/            optional intermediate CSVs read by panel scripts
```

## Locked style

Style modules under `styles/` are non-negotiable:
- `styles/deepcirc.mplstyle` — locked matplotlib rc (Arial / Helvetica / DejaVu Sans)
- `styles/colors.py` — `ACCENT_BLUE`, `PASTEL`, `FAMILY_COLORS`, `DARK_GRAY`, etc.
- `styles/typography.py` — font-size and panel-label helpers

## figtools API

```python
from figtools import use_style, save_panel, figsize_mm
from styles.colors import DARK_GRAY, ACCENT_BLUE, PASTEL, apply_axis_style

use_style()
fig, ax = plt.subplots(figsize=figsize_mm(80, 60))
# ... draw ...
save_panel(fig, "panels/vector/panel_a", dpi=600, close=True)
# writes panel_a.pdf + .svg to panels/vector/, panel_a.png to panels/raster/
```

## Build a single figure

```bash
# Build all panels for figS10:
for s in setFinal/figS10/scripts/build_panel_*.py; do python "$s"; done

# Compose the final figure from the manifest:
python scripts/build_figure.py --manifest setFinal/figS10/manifest.yaml
```

Or use the top-level Makefile:

```bash
make figures-s10
```

## Figure → data sources

| Fig | Data sources |
|---|---|
| S10 | `data/topology_g3/size_tier_designs/dense_percentile_sweep.json` + `motif_tier_analysis/structural_iso_by_size_*.csv` |
| S11 | `data/topology_g3/motif_tier_analysis/typed_expansions_by_size_*.csv` |
| S12 | `data/topology_g3/{pareto, design_predictions}/` via `_loaders.py` |
| S13 | `data/topology_g3/topology_graphs.json` + `data/topology_g3/panel_c_shapley/shapley_per_design.json` |
| S14 | `data/topology_g3/l2_top05/l2_enrichment.csv` |
| S15a | `data/topology_g3/panel_c_shapley/shapley_per_design.json` |
| S15b | `data/interp_processed/shapley_taylor_sim_{0x2B,0x17,0x6D}.json` |
