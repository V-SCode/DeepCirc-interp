# Releasing this repository to Zenodo

## Record identity

- All versions (concept DOI): [10.5281/zenodo.20576708](https://doi.org/10.5281/zenodo.20576708).
- Published v1.0.0 (version DOI): [10.5281/zenodo.20576709](https://doi.org/10.5281/zenodo.20576709).
- Use a version DOI for an exact reproducible citation. The concept DOI points to the latest published version.

The v1.0.0 record was uploaded manually. Do not assume that the GitHub integration
will continue its series. The integration follows its own previously linked
GitHub release; adding a DOI or related identifier to `.zenodo.json` does not
attach it to an existing manual series. See the [Zenodo GitHub integration FAQ](https://support.zenodo.org/help/en-gb/24-github-integration/73-can-i-pre-reserved-a-doi-before-a-github-release).

A pushed Git tag is not a published GitHub Release. Zenodo's GitHub webhook
listens for release events. Creating a release merely to retry the manual
upload can create an unrelated series if that integration is not already linked.

## Prepare a reviewable package

Use a clean, already verified checkout containing the generated setFinalvF
S10/S11 PDF and PNG previews. The packaging command checks that scientific
scripts and inputs match the tag and refuses to overwrite an existing bundle.
It does not rebuild figures, authenticate, upload, or publish.

```bash
python scripts/prepare_zenodo_release.py --ref v1.1.0 --out dist/zenodo-v1.1.0
```

It produces:

| File | Contents |
|---|---|
| `DeepCirc-interp-1.1.0.zip` | Exact tagged source, corrected intermediate data, figure scripts, and tracked S1-S3 table deliverables. |
| `DeepCirc-interp-1.1.0-setFinalvF-previews-and-tables.zip` | Existing S10/S11 composer previews (PDF/PNG), tagged S1-S3 tables (CSV/PDF/PNG/SVG), and table legends. |
| `README-ZENODO.md` | Provenance, reproduction instructions, and scope of the release. |
| `manifest.json` | Source commit, per-file provenance, sizes, SHA256 and MD5 checksums. |
| `metadata.json` | Proposed descriptive `metadata_patch`. Merge into existing draft metadata; it is not a complete API payload or an additional dataset. |

Generated S10/S11 artwork is ignored by Git and would be absent from a normal
source ZIP. The composer outputs are previews according to the figure workflow.
Final hand-composed Illustrator exports, if required for publication, must be
reviewed and added explicitly; do not rename a preview as final artwork.

The archives do not bundle the private upstream submodule or the full training
checkpoints/HDF5 datasets. The older `stage_for_zenodo.py` stages individual data
files only; it is not an uploader or a complete source/figure release packager.

The v1.1.0 tag still contains the old downloader and version-pinned v1.0.0 DOI.
Use the v1.1.0 archive's bundled data to build that version; do not overwrite it
with v1.0.0 downloads. Packaging repairs on `main` do not rewrite an existing tag.

## Resume Step C through the Zenodo website

1. Sign into the account that owns [record 20576709](https://zenodo.org/records/20576709).
2. Resume its existing unpublished new-version draft if present. Otherwise click
   **New version** on that record. Do not create a standalone new upload.
3. Check that the draft belongs to concept `20576708`, then set its version to
   `v1.1.0` and review the prepared metadata.
4. Review inherited files in the draft. Replace the old source archive with the
   new source ZIP, then add the rendering ZIP, README and manifest. The published
   v1.0.0 record remains unchanged.
5. Wait until every upload finishes. Compare filenames, byte counts and checksums
   against the manifest. Confirm the preview/final artwork distinction.
6. Review the complete record before publishing. Publication fixes that version's
   files, so complete the upload and validation first.
7. Save the new version DOI and record ID in the release audit. Update citations
   and version-pinned download defaults only after verifying the published files.

## Authenticated API recovery

Use the owner's existing authorized Zenodo session, or a personal token with
`deposit:write` and `deposit:actions` scopes. Keep tokens in a local environment
or a protected token file; do not put them in chat, Git, command-line URLs, or logs.

Follow the [official deposition API](https://developers.zenodo.org/#depositions):

1. List drafts with `GET /api/deposit/depositions?status=draft&all_versions=true&size=100`,
   following pagination if needed. Inspect only the relevant record series.
2. Resume the draft whose `conceptrecid` is `20576708` and `submitted` is `false`.
   A metadata edit of an already submitted record is not a new version.
3. Only if there is no draft, call
   `POST /api/deposit/depositions/20576709/actions/newversion`. Use the latest
   **version record ID**, not the concept ID, as this endpoint's target.
4. The new-version action returns the original resource. Follow its
   `links.latest_draft` to obtain the actual draft and its `links.bucket`.
5. Read the draft metadata and merge in the reviewed `metadata_patch` from
   `metadata.json`. Preserve its existing creators, ORCIDs, affiliations, license,
   and other bibliographic fields. The repository metadata differs from the
   published record; do not blindly replace the draft with `.zenodo.json`.
   Upload each file with a raw-byte
   `PUT` to `<bucket>/<filename>`. Use the bucket endpoint, not the deprecated
   multipart endpoint. Retrying a completed matching file is unnecessary.
6. Re-read the draft and verify sizes and checksum algorithms. Zenodo commonly
   supplies `md5:` checksums; compare MD5 with MD5, not SHA256 with MD5.
7. Publish only after checking the complete draft. Re-read the public record to
   verify the concept, version, file list and checksums, then record its DOI.

Do not use an edit of a published record to replace or append its files. File
changes require a new version. If an API call reports that a new version already
exists, locate and resume that draft rather than creating another record.

## Current recovery audit

See [the v1.1.0 recovery audit](release_audits/2026-10-07-zenodo-v1.1.0.md)
for the checked Git/Zenodo state and any remaining owner-side action.
