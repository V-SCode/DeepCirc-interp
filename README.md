# DeepCirc-interp

[![smoke](https://github.com/V-SCode/DeepCirc-interp/actions/workflows/smoke.yml/badge.svg)](https://github.com/V-SCode/DeepCirc-interp/actions/workflows/smoke.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20576708.svg)](https://doi.org/10.5281/zenodo.20576708)

End-to-end interpretability pipeline for the DeepCirc paper (Palacios et al.),
covering the cross-topology design-rule analyses and per-design attribution
work behind supplementary figures **S10–S15**.

This repository is an analysis extension to the main DeepCirc paper. It bundles the
target-function selection, topology population generation, Stage-2 MLP
training, design-space scoring, downstream design-rule analyses, and the
figure-assembly pipeline needed to reproduce S10–S15 end-to-end.

**v1.1 adds a tightened supplementary set under `figures/setFinalvF/`** (2 figures + 3 CSV/PDF tables) that fits Nature's Supplementary Information scope restriction while keeping the v1.0 six-figure set (`figures/setFinal/`) intact alongside — see [Two figure sets](#two-figure-sets-v10-vs-v11) below.

> The upstream **DeepCirc** training framework (PPO+GAT topology agent + simulator)
> lives at [sebastianrpalacios/DeepCirc](https://github.com/sebastianrpalacios/DeepCirc)
> and is vendored here as a pinned git submodule under `upstream/DeepCirc`.

## Quick links

- **Paper:** Palacios et al., *DeepCirc* (citation pending)
- **Upstream training framework:** https://github.com/sebastianrpalacios/DeepCirc
- **Zenodo archive (all versions):** [`10.5281/zenodo.20576708`](https://doi.org/10.5281/zenodo.20576708). Published v1.0.0: [`10.5281/zenodo.20576709`](https://doi.org/10.5281/zenodo.20576709). See [release status and upload instructions](docs/zenodo_release.md).
- **Working archive (internal):** `V-SCode/DeepCircMI` (private)

## What this repo contains

| Stage | Directory | What it produces |
|---|---|---|
| **A** Target selection | `topology/scripts/00_select_targets.py` + `topology/configs/target_functions.yaml` | The 20 target Boolean functions (3-input) used across G1/G2/G3. |
| **B** Topology generation | `topology/scripts/01_generate_topologies.py` + `topology/scripts/slurm/p1_ppo.sbatch` | PPO+GAT agent runs per target → `final_shared_registry.pkl`. Requires upstream DeepCirc + GPU SLURM. |
| **B'** Registry parsing | `topology/scripts/01_5_parse_registries.py` | Registry pickles → flat topology table. |
| **C** Population assembly | `topology/scripts/03_assemble_population.py` + `_population_filter.py` | Apply per-(target × size) up-to-5 sampling, agent-only filter, 4–7-reg scope. |
| **D** MLP training | `topology/scripts/04_train_mlps.py` + `slurm/p4_mlp_train*.sbatch` | Stage-2 per-(topology × task) MLPs for circuit and growth scores. |
| **E** QC + best perm | `topology/scripts/05_qc_stratify.py`, `07_5_best_perm_selection.py` | R² ≥ 0.60 circuit, R² ≥ 0.85 growth; pick best perm per topology. |
| **F** Design-space scoring | `topology/scripts/07_score_design_space.py` | Score full 20-part permutation space per retained topology. |
| **G** Cross-topology analyses | `topology/scripts/08`–`27`, `29` | L1/L2/L3 analyses feeding figS10/S11/S12/S13. |
| **H** Single-topology interp | `interp/scripts/02`–`08` | Yellow-dot Shapley + epistasis on 0x2B / 0x17 / 0x6D feeding figS14/S15. |
| **I** Figure assembly | `figures/setFinal/figS10..S15/` + `figtools/` + `styles/` | Manifest-driven Python → PDF/PNG assembly of the published figures. |

Supplementary figure → primary analysis script(s):

| Fig | Build script | Upstream analysis script(s) |
|---|---|---|
| **S10b, c** | `figures/setFinal/figS10/scripts/build_panel_{b,c}.py` | `topology/scripts/23_3_dense_percentile_sweep.py` |
| **S10d** | `figures/setFinal/figS10/scripts/build_panel_d.py` | `topology/scripts/26_l3_motif_size_aggregation.py` |
| **S11** | `figures/setFinal/figS11/scripts/build_panel_a.py` | `27_l3_motif_size_typed_expansion.py` (+ 26 prereq) |
| **S12** | `figures/setFinal/figS12/scripts/build_panel_b.py` | `14_l1_pareto_frontier.py` + `_loaders.py` |
| **S13** | `figures/setFinal/figS13/scripts/build_panel_c.py` | `22_extract_topology_graphs.py` + `29_panel_c_shapley.py` |
| **S14a, b, c** | `figures/setFinal/figS14/scripts/build_panel_{a,b,c}.py` | `09_l2_graph_role.py` |
| **S15a** | `figures/setFinal/figS15/scripts/build_panel_d.py` | `29_panel_c_shapley.py` |
| **S15b** | `figures/setFinal/figS15/scripts/build_panel_e.py` | `interp/scripts/04d_shapley_taylor_sim.py` (primary), `04b_pairwise_interactions.py` (fallback) |

## Two figure sets: v1.0 vs v1.1

v1.1 ships the Nature-SI-tightened set **alongside** the full v1.0 six-figure set. Neither replaces the other — the complete arc is useful for readers who want depth, and the tightened arc fits Nature's SI scope.

| Set | Where | Content | Status |
|---|---|---|---|
| **v1.0 full set** | [`figures/setFinal/`](figures/setFinal/) | 6 figures: S10, S11, S12, S13, S14, S15 — comprehensive analysis arc | **Frozen** — reproduces the published v1.0 DOI artifacts exactly |
| **v1.1 tightened set** | [`figures/setFinalvF/`](figures/setFinalvF/) | 2 figures (S10, S11) + 3 supplementary tables (S1, S2, S3) — Nature-SI-fit | Added in v1.1, uses the family-safe Shapley correction |

**figures/setFinalvF/figS10** consolidates panels from old-S10 (b, c, d) with old-S14 (a, b). **figures/setFinalvF/figS11** is old-S15 verbatim, re-rendered with the corrected family-safe Shapley / Shapley-Taylor attribution (see [Methods — Shapley family-uniqueness correction](#methods--shapley-family-uniqueness-correction) below). The three supplementary tables are derived from old-S13 and old-S14/c content.

Build the tightened set:
```bash
make figures-vf       # 2 figures (S10, S11)
make tables-vf        # 3 tables (S1, S2, S3)
```

## Methods — Shapley family-uniqueness correction

The Shapley and Shapley-Taylor attribution routines in `interp/scripts/04*.py` and `topology/scripts/29_panel_c_shapley.py` were updated in v1.1 to enforce DeepCirc's **family-uniqueness constraint** when drawing coalition completions: a valid 5-to-7-regulator circuit uses at most one part per TF family (e.g. only one of PhlF/P1, PhlF/P2, PhlF/P3). The pre-v1.1 implementations drew family-unique background completions from the training pool but did not re-check family uniqueness after swapping the yellow-dot parts into coalition slots, so ~49% of coalition evaluations landed on family-invalid designs (77% peak at |S|=3–4 on 0x6D 7-reg); the MLP and simulator still returned scores for those invalid designs, biasing the integrated Shapley / Shapley-Taylor values.

The v1.1 fix introduces `interp/scripts/_family_safe.py` — a precomputed per-pool family-filter sampler that returns only completions whose non-coalition slots carry no family overlap with the yellow-dot's parts at the coalition slots. All eleven swap sites across `04b_pairwise_interactions.py`, `04c_shapley_sim.py`, `04d_shapley_taylor_sim.py`, `04d_yd_vs_average.py`, `04_yellow_dot_local_analysis.py`, `04e_shapley_pairs.py`, `04e_shapley_taylor_3body_sim.py`, `04f_shapley_triples.py`, `04g_shapley_population.py`, and the valid-replacement enumeration in `29_panel_c_shapley.py` were converted to use this sampler.

Impact on the published numbers:
- **Panel A (per-design Shapley bars)**: max |ΔΦᵢ| = 971 on raw circuit score (up to 79.9% relative shift on individual slots); **0 sign-flips on circuit** across all 175 slots × 30 designs; 3 small sign-flips on growth (near-zero Φᵍ values).
- **Panel B (Shapley-Taylor Φᵢⱼ heatmap)**: max |ΔΦᵢⱼ| = 14.4 (circuit, 0x17); **0 sign-flips on circuit** across all three yellow-dots; growth-side sign-flips limited to |Φᵍ| < 0.02 values (13.3% on 0x17, 14.3% on 0x6D).

**Interpretive conclusions are preserved**: all slot-wise and pair-wise rankings on circuit survive, and the paper's "most-positive / most-destructive part at this slot" claims remain intact; the quantitative Φ values shift and the published `figures/setFinalvF/figS11/final/figure_S11.pdf` reflects the corrected numbers.

For reproducibility, the pre-correction (v1.0) outputs are preserved under `data/interp_processed/_pre_family_fix/` and `data/topology_g3/panel_c_shapley/_pre_family_fix/`; the v1.0 `figures/setFinal/figS15` build scripts are pinned to read from those backups so `make figures-s15` reproduces the exact v1.0 FigS15.ai numbers.

## Installation

```bash
git clone --recurse-submodules https://github.com/V-SCode/DeepCirc-interp.git
cd DeepCirc-interp

# Option 1 — conda (recommended)
conda env create -f environment.yml
conda activate deepcirc-interp

# Option 2 — pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Install this package in editable mode so `from topology.X import ...`,
# `from interp.X import ...`, and `from figures.X import ...` resolve.
pip install -e .

# Install the upstream DeepCirc training framework (vendored submodule).
# Required for Path 2 full re-runs (PPO+GAT topology generation, MLP training,
# simulator-valued Shapley-Taylor); not needed for Path 1 figure rebuilds.
pip install -e upstream/DeepCirc
```

## Two reproducibility paths

### Path 1 — Rebuilding the figures (laptop, ~minutes)

Use the pre-computed intermediates bundled in this checkout or its versioned
source archive and run the figure-assembly pipeline. No GPU, no SLURM, no training.

```bash
# A checkout/source ZIP already includes the small figure-input data.
# Optional: restore data from the version-pinned Zenodo archive (currently v1.0.0).
# Do not replace v1.1 corrected data with v1.0 data.
# python scripts/download_data.py --tier figures
make figures
# Outputs land in figures/setFinal/figS{10..15}/final/
```

### Path 2 — End-to-end interpretability (HPC, days to weeks depending on compute access)

Re-execute the entire end-to-end pipeline including training. Requires a
SLURM-managed GPU cluster (developed and tested on MIT Engaging / ORCD;
SLURM templates under `topology/scripts/slurm/`).

```bash
# A. Target selection
python topology/scripts/00_select_targets.py

# B. Topology generation (PPO+GAT, GPU, ~hours per target × 20)
cd topology/scripts/slurm && sbatch --array=0-19 p1_ppo.sbatch

# C. Population assembly
python topology/scripts/03_assemble_population.py

# D. MLP training (~hours per topology × 215, parallelizable)
sbatch --array=0-214 p4_mlp_train.sbatch

# E. QC + best-perm selection
python topology/scripts/05_qc_stratify.py
python topology/scripts/07_5_best_perm_selection.py

# F. Design-space scoring
python topology/scripts/07_score_design_space.py

# G. Cross-topology analyses
make analyses

# H. Single-topology interp (per-exemplar Shapley, ~hours)
make interp

# I. Figures
make figures
```

Intermediate Path-2 outputs at each stage are checkpointed against Zenodo
artifacts so reviewers can re-enter the pipeline at any phase.

## Repository layout

```
DeepCirc-interp/
├── README.md                 ← you are here
├── LICENSE                   ← MIT
├── environment.yml           ← conda (recommended)
├── requirements.txt          ← pip-pinned
├── Makefile                  ← end-to-end orchestration
├── .gitmodules               ← pins upstream DeepCirc commit
│
├── upstream/                 ← git submodule → sebastianrpalacios/DeepCirc
│   └── DeepCirc/             (PPO+GAT agent + simulator + libs)
│
├── topology/                 ← cross-topology pipeline (figS10–S13)
│   ├── configs/              ← target_functions.yaml, pipeline.yaml
│   ├── scripts/              ← P0–P29 analysis pipeline
│   │   └── slurm/            ← SLURM templates for cluster runs
│   └── figures/              ← shared figure renderers (fig18/29/30/34v4)
│
├── interp/                   ← single-topology interp (figS14–S15)
│   └── scripts/              ← loaders + simulator + 02–08 modules
│
├── figures/                  ← paper-figure assembly pipeline
│   ├── figtools/             ← export, layout, validate
│   ├── styles/               ← colors, typography, mplstyle
│   └── setFinal/figS10..S15/ ← per-figure scripts + manifests
│
├── data/                     ← inputs (mostly fetched from Zenodo)
│   ├── exemplars/            ← 0x2B / 0x17 / 0x6D yellow-dot data
│   ├── topology_g3/          ← G3 substrate predictions + analyses
│   └── interp_processed/     ← per-exemplar Shapley + epistasis JSONs
│
├── docs/                     ← supplementary documentation
│   └── composition_paragraph.md
│
└── scripts/                  ← top-level utilities (download_data.py, etc.)
```

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `DEEPCIRC_EXEMPLARS` | `./data/exemplars/` | Where `interp/scripts/loaders.py` looks for per-exemplar HDF5 / MLP checkpoint directories (`{0x2B,0x17,0x6D}_design/`). |
| `DEEPCIRC_DATA` | `./data/` | Root for downloaded Zenodo intermediates. |

## Citation

If you use this repository, please cite the DeepCirc paper:

```bibtex
@article{palacios2026deepcirc,
  title  = {DeepCirc: ...},
  author = {Palacios, Sebastian R. and ...},
  year   = {2026},
  ...
}
```

And the Zenodo deposit for this companion repository:

```bibtex
@software{vege2026deepcirc_interp,
  title     = {DeepCirc-interp: end-to-end interpretability companion code for the DeepCirc paper},
  author    = {Vege, Venkat},
  year      = {2026},
  publisher = {Zenodo},
  version   = {v1.0.0},
  doi       = {10.5281/zenodo.20576709},
  url       = {https://doi.org/10.5281/zenodo.20576709}
}
```

## License

MIT (see [LICENSE](LICENSE)). The vendored upstream DeepCirc submodule under
`upstream/DeepCirc/` is governed by its own license.

## Tested on

- macOS 15 (Darwin 25.3), Python 3.12.13, conda env from `environment.yml`
- ubuntu-latest GitHub Actions runner, Python 3.10 / 3.11 / 3.12 via pip
- All six supplementary figures (S10–S15) regenerate from Tier 0 in-repo data
  alone — see [docs/smoke_test_results.md](docs/smoke_test_results.md)

## Acknowledgments

This work builds on the upstream DeepCirc framework by Sebastian Palacios and
the Jim Collins lab.
