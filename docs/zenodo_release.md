# Releasing this repository to Zenodo

This repo carries a two-tier Zenodo deposit:

- **`figures` tier** — small CSV/JSON intermediates for laptop figure rebuilds (~10 MB)
- **`full` tier** — trained MLP checkpoints, design-space predictions, exemplar HDF5s for full re-runs (~tens of GB)

The repo also ships [.zenodo.json](../.zenodo.json) for auto-population of the
deposit metadata via the GitHub → Zenodo integration.

## One-time setup (do this before the first release)

1. Sign in to [Zenodo](https://zenodo.org/) with your ORCID or GitHub.
2. Go to [Zenodo → GitHub integration](https://zenodo.org/account/settings/github/)
   and **flip the toggle next to `V-SCode/DeepCirc-interp` to ON.**
3. From this point forward, every GitHub Release tagged `v*` on the repo will
   automatically mint a new Zenodo DOI and pull the tag's source archive +
   `.zenodo.json` metadata into the deposit.

## Releasing the `figures` tier

The figures tier is everything currently committed to git (~10 MB after Phase 4
+ any tier-1 additions). It ships automatically with the GitHub Release source
archive, so the figures tier mints alongside the code DOI:

```bash
cd /path/to/DeepCirc-interp
git tag -a v1.0.0 -m "DeepCirc-interp v1.0.0 — figures tier"
git push origin v1.0.0
gh release create v1.0.0 --title "v1.0.0 — figures tier" \
    --notes "Companion repo for DeepCirc paper supplementary figures S10–S15."
# Wait ~1 minute; check https://zenodo.org/account/settings/github/ for the
# minted DOI.
```

## Releasing the `full` tier

The full tier requires uploading the large back-end artifacts (MLP
checkpoints, design-space predictions, exemplar HDF5s) as separate files on
the Zenodo record. The GitHub → Zenodo integration **only attaches the source
archive**; large artifacts must be added manually.

Two approaches:

### Approach 1 — Manual upload via Zenodo UI

```bash
# Stage the full-tier bundle locally
python scripts/stage_for_zenodo.py --tier full --out ./zenodo_stage_full
# Required: data/topology_g3/{registries,mlp_checkpoints,population.pkl,qc_tiers.pkl}
# Required: data/exemplars/0x{2B,17,6D}_design/
# These must be rsync'd from the cluster scratch dir first; see
# `topology/scripts/slurm/README.md`.
```

Then open the figures-tier deposit on Zenodo, edit, and upload each file in
`./zenodo_stage_full/` via the web UI. Tag the new files with `full__` prefix
(already done by `stage_for_zenodo.py`).

### Approach 2 — zenodo-client CLI

```bash
pip install zenodo-client
# Get your token at https://zenodo.org/account/settings/applications/tokens/new/
export ZENODO_API_TOKEN="<your token>"

# Stage as above, then:
for f in ./zenodo_stage_full/full__*; do
    zenodo-client upload --deposition-id <id-from-figures-tier-DOI> "$f"
done
```

## Updating the deposit

Zenodo records are versioned. To publish a new version (e.g., after a bug fix):

```bash
git tag -a v1.0.1 -m "DeepCirc-interp v1.0.1 — fix XYZ"
git push origin v1.0.1
gh release create v1.0.1 ...
```

The Zenodo integration mints a new DOI but preserves the concept DOI (the
"all-versions" DOI you cite in the paper, which always redirects to the latest
version).

## Citing the deposit in the paper

After the first release, replace the `[reference / DOI]` placeholder in the
paper Methods section "Data and code availability" subsection with:

> Code and intermediate data for the cross-topology interpretability analyses
> and supplementary figures S10–S15 are available at
> `https://github.com/V-SCode/DeepCirc-interp` and archived at Zenodo
> (DOI: [`10.5281/zenodo.20576709`](https://doi.org/10.5281/zenodo.20576709)).

Use the **concept DOI** (not the version DOI) so the citation tracks future
updates.
