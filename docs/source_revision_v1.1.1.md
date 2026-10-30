# v1.1.1 source correction status

Prepared on 2026-10-30 from research commit `0556a8bb137b03f67951cdca93d0161aed4565f4` in V-SCode/DeepCircMI. This companion revision is a code/data/source-panel correction. It does not constitute a new final Illustrator figure set or an updated manuscript.

## Included

- Versioned circuit-presence data in `figures/setFinalvF/data/`, with exact recovered context denominators, input hashes, null-space identifiability checks and comparison with the historical count statistic. Original G3 data remain unchanged.
- S10d/e and S3 use `log2[((n_top+1)/(N_top+2))/((n_bottom+1)/(N_bottom+2))]`. Each evaluated assignment row is counted once per context per tier. Sampled assignment rows are not deduplicated. Original inclusive percentile masks and all cutoff ties are retained.
- S10a/b retain their within-size percentile distributions; S10c retains the motif-presence rate analysis with a corrected title.
- S11 retains the corrected numerical attribution inputs. Panel a shows single-slot replacement contributions normalized by their within-design absolute sum. Panel b shows pairwise Shapley interaction estimates on raw c/g and conventional Shapley values in the adjacent stripe. One small negative growth pair is preserved visibly. Growth colours saturate outside the shared ±0.05 scale; labels retain numerical values.
- The attribution selection generator now uses log2/exp2 consistently with stored Pareto inputs. Tests confirm that the selected 30 entries are unchanged.
- Every source panel has a rebuild path. A strict preview entry point rejects missing vector/raster panels, so a missing-panel placeholder can no longer pass as a complete current preview.

## Deliberately deferred

Final Illustrator assembly and root `FigSNNvF.ai/.pdf/.png` publication masters will be regenerated and reviewed by the user. Paper text, Methods, figure legends and table legends are not updated in this revision. Any existing `table_legends.md` or historical explanatory material retained in the source archive must not be used as a caption for the new normalized data. The rendering archive excludes standalone legend files and stale final masters. Historical `figures/setFinal/` scripts and G3 inputs are preserved.

## Reproduce and validate

```sh
python -m pip install -r requirements.txt
python -m pip install -e .
make test-vf
make presence-data-vf    # optional exact regeneration of bundled current data
make figures-vf tables-vf
```

Arial must be available for the established table/figure typography. The native-margin helper adds page space without scaling source vector artwork. Generated `publication/config.*` paths are local to the current checkout and are regenerated, not distributed as portable absolute paths.

The current source archive includes the small analysis inputs and corrected tables, not the private upstream submodule or full model/HDF5 datasets. Figure rebuilding and tests require no model or simulator rerun.

## Release identity

The confirmed previous published version is v1.1.0, record `23227964`, DOI `10.5281/zenodo.23227964`. The concept remains `10.5281/zenodo.20576708`. The record title remains **DeepCirc-Interp: Data Files and Code**. A v1.1.1 package prepared before commit/tag is explicitly a working-tree review bundle, with base-commit and source-file hashes. After review, rebuild from the new immutable tag. No tag, commit, GitHub Release, upload or publication is performed by the package preparation command.
