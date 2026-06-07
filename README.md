# DeepCirc-interp

[![smoke](https://github.com/V-SCode/DeepCirc-interp/actions/workflows/smoke.yml/badge.svg)](https://github.com/V-SCode/DeepCirc-interp/actions/workflows/smoke.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20576207.svg)](https://doi.org/10.5281/zenodo.20576207)

End-to-end interpretability pipeline for the DeepCirc paper (Palacios et al.),
covering the cross-topology design-rule analyses and per-design attribution
work behind supplementary figures **S10–S15**.

This repository is the companion to the main DeepCirc paper. It bundles the
target-function selection, topology population generation, Stage-2 MLP
training, design-space scoring, downstream design-rule analyses, and the
figure-assembly pipeline needed to reproduce S10–S15 end-to-end.

> The upstream **DeepCirc** training framework (PPO+GAT topology agent + simulator)
> lives at [sebastianrpalacios/DeepCirc](https://github.com/sebastianrpalacios/DeepCirc)
> and is vendored here as a pinned git submodule under `upstream/DeepCirc`.

## Quick links

- **Paper:** Palacios et al., *DeepCirc* (citation pending)
- **Upstream training framework:** https://github.com/sebastianrpalacios/DeepCirc
- **Zenodo deposit (code archive + data tiers):** [`10.5281/zenodo.20576207`](https://doi.org/10.5281/zenodo.20576207)
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

# Install the upstream DeepCirc training framework (vendored submodule)
pip install -e upstream/DeepCirc
```

## Two reproducibility paths

### Path 1 — "Just rebuild the figures" (laptop, ~minutes)

Download pre-computed intermediates from Zenodo and run the figure-assembly
pipeline only. No GPU, no SLURM, no training.

```bash
python scripts/download_data.py --tier figures   # ~10 MB
make figures
# Outputs land in figures/setFinal/figS{10..15}/final/
```

### Path 2 — "Full re-run from scratch" (HPC, days to weeks)

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
  doi       = {10.5281/zenodo.20576207},
  url       = {https://doi.org/10.5281/zenodo.20576207}
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
the Collins lab.

The repository structure, migration tooling, and documentation in this
companion repo were assembled with support from
[Claude Code](https://claude.com/claude-code).
