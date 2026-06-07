# interp/

Single-topology mechanistic-interpretability pipeline on the three Fig. 3B
yellow-dot exemplars (0x2B / 0x17 / 0x6D). Produces per-design Shapley and
pairwise Shapley-Taylor epistasis used in supplementary figures **S14a/b/c**
(via `topology/scripts/09_l2_graph_role.py` aggregated context) and **S15a/b**.

## Pipeline

| Stage | Script | Output |
|---|---|---|
| Loader smoke test | `scripts/01_loaders.py` | Sanity-check on exemplar I/O |
| Master tables | `scripts/02_make_master_tables.py` | `data/interp_processed/master_{0x2B,0x17,0x6D}.parquet` |
| Landscape audit | `scripts/03_landscape_audit.py` | Predicted-vs-actual, Pareto, top-k per exemplar |
| Slot Shapley (local) | `scripts/04_yellow_dot_local_analysis.py` | On-manifold slot Shapley + counterfactuals |
| Pairwise (MLP) | `scripts/04b_pairwise_interactions.py` | `data/interp_processed/pairwise_{0x2B,0x17,0x6D}.json` |
| Shapley-Taylor (simulator) | `scripts/04d_shapley_taylor_sim.py` | `data/interp_processed/shapley_taylor_sim_{0x2B,0x17,0x6D}.json` |
| Good/bad comparisons | `scripts/05_good_bad_comparisons.py` | Module C output |
| Global decomposition | `scripts/06_global_decomposition.py` | ANOVA-style main + pairwise (Module D) |
| Stagewise audit | `scripts/07_stagewise_audit.py` | Selection-pipeline funnel (Module G) |
| Model-internal | `scripts/08_model_internal_analysis.py` | Layerwise probes (Module F) |

## Configuration

Exemplar paths are resolved by `scripts/loaders.py` from the `DEEPCIRC_EXEMPLARS`
environment variable (default: `./data/exemplars/`). Each exemplar directory
must contain the HDF5 valid-permutation samples and MLP checkpoints for that
target. Layout matches upstream DeepCirc `output_folder_name`:

```
data/exemplars/
├── 0x2B_design/
│   ├── sampled_valid_permutations_5_gates_*.h5
│   ├── circuit_score_model.pt
│   └── toxicity_score_model.pt
├── 0x17_design/...
└── 0x6D_design/...
```

## Methodological notes

- **Slot-level Shapley, not vanilla SHAP.** The valid assignment space is
  constrained by one-hot slot structure and the no-repeated-protein rule.
  Independent feature masking creates invalid off-manifold inputs. We use
  slot-level Shapley with the value function averaging over valid completions.
- **`toxicity_score` on disk = paper's `growth_score`.** Loader exposes both names.
- **Slot order = regulator-node-index order** in the networkx DAG (confirmed via
  yellow-dot cross-reference for 0x2B; MLP round-trip on saved predictions
  verifies for 0x17 and 0x6D).
