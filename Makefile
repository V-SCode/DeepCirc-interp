# DeepCirc-interp — end-to-end reproducibility orchestration.
#
# Two paths:
#   PATH 1 — Just rebuild the figures (laptop, minutes):
#     make data-figures && make figures
#
#   PATH 2 — Full re-run from scratch (HPC, days to weeks):
#     make targets && make topologies && make population && \
#     make mlps && make qc && make scoring && make analyses && \
#     make interp && make figures
#
# Variables (override on the command line):
PY         ?= python
DATA_ROOT  ?= ./data
ZENODO_DOI ?= 10.XXXX/XXXXX
FIG_OUT    := figures/setFinal

.PHONY: help env data data-figures data-full \
        targets topologies population mlps qc scoring \
        analyses interp figures figures-s10 figures-s11 figures-s12 \
        figures-s13 figures-s14 figures-s15 clean clean-figures

help:
	@echo "DeepCirc-interp Makefile targets:"
	@echo ""
	@echo "  Environment setup:"
	@echo "    env              Install conda env from environment.yml"
	@echo ""
	@echo "  Data:"
	@echo "    data-figures     Fetch ~10 MB of figure-input intermediates from Zenodo (Path 1)"
	@echo "    data-full        Fetch all back-end artifacts incl. trained MLPs and predictions (~tens of GB)"
	@echo ""
	@echo "  Pipeline (Path 2 — full re-run from scratch):"
	@echo "    targets          Phase A: select 20 target Boolean functions"
	@echo "    topologies       Phase B: PPO+GAT topology generation (SLURM)"
	@echo "    population       Phase C: assemble + filter population"
	@echo "    mlps             Phase D: train Stage-2 MLPs (SLURM)"
	@echo "    qc               Phase E: QC stratify + best-perm select"
	@echo "    scoring          Phase F: score full design space"
	@echo "    analyses         Phase G: L1/L2/L3 cross-topology analyses"
	@echo "    interp           Phase H: single-topology Shapley + epistasis"
	@echo ""
	@echo "  Figure assembly (Phase I):"
	@echo "    figures          Build all S10–S15 figures"
	@echo "    figures-s10..s15 Build a single figure"
	@echo ""
	@echo "  Cleanup:"
	@echo "    clean            Remove all generated artifacts"
	@echo "    clean-figures    Remove only figure outputs"

# --- Environment ---

env:
	conda env create -f environment.yml
	@echo "Activate with: conda activate deepcirc-interp"
	@echo "Then install upstream submodule: pip install -e upstream/DeepCirc"

# --- Data fetching ---

data-figures:
	$(PY) scripts/download_data.py --tier figures --root $(DATA_ROOT) --doi $(ZENODO_DOI)

data-full:
	$(PY) scripts/download_data.py --tier full --root $(DATA_ROOT) --doi $(ZENODO_DOI)

data: data-figures

# --- Path 2: full re-run from scratch ---

targets:
	$(PY) topology/scripts/00_select_targets.py

topologies:
	@echo "Submit PPO+GAT topology generation via SLURM:"
	@echo "  cd topology/scripts/slurm && sbatch --array=0-19 p1_ppo.sbatch"
	@echo "Then parse registries:"
	@echo "  $(PY) topology/scripts/01_5_parse_registries.py"

population:
	$(PY) topology/scripts/03_assemble_population.py

mlps:
	@echo "Submit Stage-2 MLP training via SLURM:"
	@echo "  cd topology/scripts/slurm && sbatch --array=0-214 p4_mlp_train.sbatch"

qc:
	$(PY) topology/scripts/05_qc_stratify.py
	$(PY) topology/scripts/07_5_best_perm_selection.py

scoring:
	$(PY) topology/scripts/07_score_design_space.py

analyses:
	$(PY) topology/scripts/22_extract_topology_graphs.py
	$(PY) topology/scripts/09_l2_graph_role.py
	$(PY) topology/scripts/23_2_size_tier_design_counts.py
	$(PY) topology/scripts/23_3_dense_percentile_sweep.py
	$(PY) topology/scripts/24_l3_motif_tier_aggregation.py
	$(PY) topology/scripts/26_l3_motif_size_aggregation.py
	$(PY) topology/scripts/27_l3_motif_size_typed_expansion.py
	$(PY) topology/scripts/29_panel_c_shapley.py

interp:
	$(PY) interp/scripts/02_make_master_tables.py
	$(PY) interp/scripts/03_landscape_audit.py
	$(PY) interp/scripts/04_yellow_dot_local_analysis.py
	$(PY) interp/scripts/04d_shapley_taylor_sim.py
	$(PY) interp/scripts/04b_pairwise_interactions.py
	$(PY) interp/scripts/05_good_bad_comparisons.py
	$(PY) interp/scripts/06_global_decomposition.py
	$(PY) interp/scripts/07_stagewise_audit.py
	$(PY) interp/scripts/08_model_internal_analysis.py

# --- Figure assembly (Phase I) ---

figures: figures-s10 figures-s11 figures-s12 figures-s13 figures-s14 figures-s15

define BUILD_FIG
	for s in $(FIG_OUT)/$(1)/scripts/build_panel_*.py; do $(PY) "$$s"; done
	$(PY) figures/scripts/build_figure.py --manifest $(FIG_OUT)/$(1)/manifest.yaml
endef

figures-s10: ; $(call BUILD_FIG,figS10)
figures-s11: ; $(call BUILD_FIG,figS11)
figures-s12: ; $(call BUILD_FIG,figS12)
figures-s13: ; $(call BUILD_FIG,figS13)
figures-s14: ; $(call BUILD_FIG,figS14)
figures-s15: ; $(call BUILD_FIG,figS15)

# --- Cleanup ---

clean: clean-figures
	rm -rf data/topology_g3/predictions data/topology_g3/mlp_checkpoints
	rm -rf data/interp_processed/master_*.parquet

clean-figures:
	rm -f $(FIG_OUT)/*/panels/raster/*.png
	rm -f $(FIG_OUT)/*/panels/vector/*.pdf $(FIG_OUT)/*/panels/vector/*.svg
	rm -f $(FIG_OUT)/*/final/*.pdf $(FIG_OUT)/*/final/*.png
	rm -f $(FIG_OUT)/*/final/*.linked_assets.json
