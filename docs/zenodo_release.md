# Preparing the v1.1.1 correction release

Published by the user on 2026-10-30 as
[10.5281/zenodo.23822701](https://doi.org/10.5281/zenodo.23822701), from
tag `v1.1.1` at commit `3761887960672f63a252feff29a8ee0c41ed930c`.
All five published file sizes and MD5 checksums were verified after
publication. [The release receipt](release_v1.1.1.json) records the checks
and successful CI; the preparation procedure below is retained for reference.

This is a source-panel and data correction. Illustrator master regeneration
and all manuscript text, Methods, figure legends and table legends are deferred.
No Word document or existing Illustrator master belongs in this upload.

The published baseline is v1.1.0, [record 23227964](https://zenodo.org/records/23227964).
The all-versions concept DOI is [10.5281/zenodo.20576708](https://doi.org/10.5281/zenodo.20576708).
Record 20576709 is the older v1.0.0 version, not the concept record.
The public title remains **DeepCirc-Interp: Data Files and Code**.

The research source for this correction is `V-SCode/DeepCircMI` commit
`0556a8bb137b03f67951cdca93d0161aed4565f4`. The companion ports the scientific
changes into its portable paths. Its own reviewed source commit is recorded
separately in each tagged package.

## Build and review locally

Install the repository requirements and editable package, including Arial for
the intended typography, then build the complete previews and tables:

```bash
make figures-vf tables-vf
python -m unittest discover -s tests -p test_prepare_zenodo_release.py -v
```

To inspect an uncommitted candidate without inventing a release commit or tag:

```bash
python scripts/prepare_zenodo_release.py --working-tree --version v1.1.1 \
  --out dist/zenodo-v1.1.1-review
```

This mode uses an explicit source-file allowlist over Git's tracked and
nonignored untracked files. It excludes credentials, runtime files, private
submodule contents, resource forks, generated source panels, and stale
Illustrator masters from the source snapshot. Generated artwork enters only
through the separate explicit rendering roster. The manifest sets
`source_tag` and `source_commit` to null, records the base commit and every
file's hashes, and marks the bundle **not ready for upload**.

After the reviewed changes have been committed and tagged separately, prepare
the final bundle from the clean tagged source:

```bash
python scripts/prepare_zenodo_release.py --ref v1.1.1 \
  --out dist/zenodo-v1.1.1
```

Tagged mode creates an exact `git archive` of the tag and rejects any difference
between the working source and that archive, including new untracked source
files. It also refuses excluded runtime/credential/artwork files in a tag.
Neither packaging mode overwrites an existing output directory. Neither
command rebuilds figures, authenticates, uploads, tags, commits, or publishes.

## Package contents and checks

| Outer file | Contents |
|---|---|
| `DeepCirc-interp-1.1.1.zip` | Source, tests, small intermediate inputs, normalized presence data, manifests and tracked table deliverables. |
| `DeepCirc-interp-1.1.1-setFinalvF-source-panels-and-tables.zip` | Seven source panels in PDF/SVG/PNG, four prepared native-scale placement pages in PDF/PNG, complete S10/S11 composer previews in PDF/PNG, two placement manifests, and three tables in CSV/PDF/PNG/SVG. |
| `README-ZENODO.md` | Exact source provenance, scope, rebuilding instructions and deferred work. |
| `metadata.json` | Descriptive `metadata_patch` only; it is not a complete deposition payload. |
| `manifest.json` | Source and artifact provenance, byte counts, MD5 and SHA256 hashes, and deferred-work status. |

The artwork archive has 47 required assets plus its README. All seven source
panels are included even when a panel's numerical values did not change.
The packager requires every vector and raster asset referenced by each
manifest, rejects incomplete panel rosters and missing-preview placeholders,
parses PDFs/SVGs/PNGs/CSVs, and checks ZIP CRCs. Table and manifest bytes must
agree between the source and artwork archives. It detects concurrent changes
to source or artwork before writing the bundle.

S10d/e and Table S3 use normalized circuit-presence rates under the original
inclusive percentile masks, including cutoff ties. The actual denominator is
the number of evaluated circuits containing each structural context in that
tier. S11 attribution data are unchanged; its units, terminology, signed-value
display and source validation are corrected. See the scientific audit files
for the full calculation and provenance.

Composer PDFs/PNGs are explicitly **previews**, not final Illustrator exports.
The source may retain unchanged historical legend files for provenance; these
do not describe the corrected statistic and are not current manuscript text.
Standalone legends are omitted from the artwork archive. Do not use the old
legend files as current descriptions of these panels.

The source archive excludes the private upstream submodule and large training
checkpoints/HDF5 data. It is not a complete full-data tier. Use its bundled
small inputs for this version instead of replacing them with older downloads.

## Continue the existing Zenodo series

The original record series was uploaded manually. A pushed Git tag does not
publish a Zenodo version. Do not create a GitHub Release merely to trigger an
unverified webhook connection; the integration can follow a different series.
See the [Zenodo GitHub integration guidance](https://support.zenodo.org/help/en-gb/24-github-integration/73-can-i-pre-reserved-a-doi-before-a-github-release).

Use an existing authorized owner session or a securely stored token with
`deposit:write` and `deposit:actions`. Never put tokens in chat, Git, command-line
URLs, ZIPs or logs. The packager neither reads nor needs credentials.

For the [deposition API](https://developers.zenodo.org/#depositions):

1. Confirm the latest published version and concept identity. List unpublished
   drafts with `GET /api/deposit/depositions?status=draft&all_versions=true&size=100`,
   following pagination, and resume the draft for concept `20576708` if one exists.
2. Only when there is no draft, request a new version from the latest version
   record, currently `POST /api/deposit/depositions/23227964/actions/newversion`.
   Follow `links.latest_draft` in the response. Do not target the concept record
   or create a new standalone deposition.
3. Read existing metadata and merge the reviewed `metadata_patch`. Preserve
   creators, ORCIDs, affiliations, license, and other bibliographic fields;
   do not replace the metadata wholesale with `.zenodo.json`.
4. Replace inherited obsolete release files only in the unpublished draft.
   Upload the five files with raw-byte `PUT` requests to its `links.bucket`.
5. Re-read the draft. Verify every filename, byte count and MD5 against the
   local artifacts. Compute the final manifest's MD5 directly because it cannot
   contain its own checksum. Compare identical algorithms; a SHA256 value is
   not comparable to Zenodo's MD5. Do not upload an untagged review snapshot.
6. Review and publish only when separately authorized. Verify the public record
   and record its new version DOI and checksum result in a release audit.

Published version files are immutable: changed files require a new version.
If Zenodo reports that a draft already exists, resume it rather than creating
duplicates. The [2026-10-07 audit](release_audits/2026-10-07-zenodo-v1.1.0.md)
is a historical prepublication recovery record, not the current release state.
