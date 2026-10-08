# setFinalvF — tightened / consolidated supplementary figure set

Iterative v2 of the [setFinal/](../setFinal/) supplementary figure set for the DeepCirc paper. Purpose: tighten and consolidate the S10–S15 arc, including converting two of the figures to tables.

**Distribution model is additive, not replacement.** setFinal/ stays frozen as the v1.0 deliverable already published to DeepCirc-interp + Zenodo (v1.0.0 version DOI [`10.5281/zenodo.20576709`](https://doi.org/10.5281/zenodo.20576709); all-versions DOI [`10.5281/zenodo.20576708`](https://doi.org/10.5281/zenodo.20576708)); setFinalvF/ ships alongside as a new version DOI under the same concept.

## Per-figure structure

Mirrors setFinal/ exactly:

```
figSNN/
  _spec/             hand-drawn sketches + notes
  data/              intermediate CSV/parquet for panel scripts
  scripts/           build_panel_*.py
  panels/
    vector/          panel_*.pdf + .svg
    raster/          panel_*.png
  final/             composer output (preview only, not shipped)
  manifest.yaml      mm-precise panel layout
```

Shipped `.ai` / `.pdf` / `.png` deliverables sit at this folder's root as `FigSNNvF.{ai,pdf,png}`.

## Workflow

Same Python-first, Illustrator-last workflow as setFinal/:

1. Draft in Python (panel PDFs + PNGs) — iterate here, review PNGs.
2. Hand-compose in Adobe Illustrator per [../../illustrator/README_setFinal_workflow.md](../../illustrator/README_setFinal_workflow.md).
3. Export `FigSNNvF.ai` + `FigSNNvF.pdf` + `FigSNNvF.png`.

## Roster

### Figures

| Folder | Content | Layout | v1 lineage |
|---|---|---|---|
| [figS10/](figS10/) | Design-space + structural motifs + per-part landscapes | 5 panels (a/b left-col, c right-col, d/e bottom-row), 240×302 mm | old S10 b/c/d + old S14 a/b |
| [figS11/](figS11/) | Per-design Shapley + pairwise epistasis on yellow-dots | 2 panels (a top, b bottom), 370×490 mm | old S15 (unchanged) |

### Supplementary Tables

| Folder | Content | Shape | v1 lineage |
|---|---|---|---|
| [tabS01/](tabS01/) | Top 15 max-circuit designs (one per target function), ranked by circuit score | 15 rows × 6 cols | old S13 top-right table |
| [tabS02/](tabS02/) | Top 15 Pareto-knee designs (one per target), ranked by combined score | 15 rows × 7 cols | old S13 bottom-right table |
| [tabS03/](tabS03/) | Mean log₂ enrichment per part × cascade position bucket | 20 rows × 6 cols | old S14 panel c heatmap |

The CSVs (`final/table_SN.csv`) are the authoritative data for import into Word,
LaTeX, Docs, or DeepCirc-interp. Each table also has polished `PDF`, editable
`SVG`, and 600-dpi white-background `RGB PNG` versions in the same `final/`
directory. The renderers read the CSVs without changing values or row order.

Regenerate the styled tables from the repo root:

```bash
/opt/miniconda3/bin/python paper_figures/figures/setFinalvF/tabS01/scripts/render_table.py
/opt/miniconda3/bin/python paper_figures/figures/setFinalvF/tabS02/scripts/render_table.py
/opt/miniconda3/bin/python paper_figures/figures/setFinalvF/tabS03/scripts/render_table.py
```

S1/S2 share `_shared/design_tables.py`: Arial, pale blue headers, thin gray
rules, measured part-name spacing, and the categorical part colors of the
delivered FigS13v2/FigS15 figures. S1 is 230 × 110 mm; S2 is 230 × 157 mm.
Their displayed circuit/growth scores are raw MLP predictions; S2's combined
score uses normalized log₂ circuit score plus normalized growth score.

S3 is 180 × 136 mm with grouped NOT/NOR headers and a pale gray–white–blue scale centered on
zero. The color scale saturates at ±3; all numerical values remain visible.
Uniform dark row labels leave color to encode the enrichment values alone.

All artwork contains only column headers and data, in Arial. Titles,
definitions, part-color explanations, S2 count denominators, and S3's scale
and outlier note are separate in [table_legends.md](table_legends.md), for
placement in the manuscript's table legends.

## Illustrator handoff

See [../../illustrator/README_setFinalvF_workflow.md](../../illustrator/README_setFinalvF_workflow.md) for the full per-figure spec table, panel-placement coordinates, label sizes, and export presets. Shipped deliverables (`FigSNNvF.ai/.pdf/.png`) live at this folder's root.
