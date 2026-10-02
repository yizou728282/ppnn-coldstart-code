# Reproducibility of the publication snapshot

## Three levels

1. **Manuscript and submission package:** not included in this repository.
2. **Inspection and summary replay:** protocols, aggregate CSV/JSON results, numerical provenance and original audit checks are public.
3. **Independent complete recomputation:** requires original provider data plus aligned per-seed predictions. Raw observations and processed observation metadata are not redistributed. Preserved predictions are now public Release assets, with SHA-256 manifests and observation-free row indexes; see PREDICTIONS.md at the repository root. Complete independent replay still requires rebuilding observation metadata from provider data, matching index order and preparing the original evaluation/audit workspace. This publication step did not execute that replay.

## Original numerical audit

The 2026-09-29 audit recovered original Stage-2/3/4 predictions and combined them with Stage-5 outputs. It reports 70 score combinations, original calibration and daily comparisons, 1,986 matching score/test fields, and 112 matching primary DRN/GNN DiD fields including six additional spatial partitions. Audit output checks and executable code are preserved under `handoff/original_prediction_audit/`. These records are provenance of the existing audit, not a newly executed audit in this publication preparation.

The full auxiliary pair/ranking/fold-block/sensitivity suite and all plots were not regenerated in that audit. The later matched-seed, forecast-only and regional-block DiD intervals (`results/review_supplementary/`) reproduce the primary DiD intervals in `results/stage5/eval/final/decisions.json` exactly before adding new contrasts. Numerical agreement does not resolve station dependence, unmatched predictors, seen-station validation, unequal ensemble sizes or overlapping additional partitions.

## Scope of this snapshot

The manuscript and the journal submission package are not in this repository. No experiments, model settings or existing result-table values were changed for this export. The 2026-10-01 reanalysis added post hoc DiD intervals from the stored predictions (`analysis/`, `results/review_supplementary/`).

## Internal registration history

Protocol analysis definitions, dates and decision rules are retained. PUBLIC_FILES.json preserves the source inventory and wording-edit record of the earlier publication snapshot; it is a historical provenance record rather than an inventory of this repository. Paths and file identities for the frozen audit sources and saved table exports are recorded in `provenance/file_origins.json`. The publication repository starts with a new history and does not independently expose the original development commit chronology. Internal registration must not be represented as external preregistration.

The exact source versions used by the 2026-09-29 audit are included in `handoff/original_prediction_audit/source/`, with SHA-256 checksums. The audit loaders can use these files without retrieving the historical source commit. Provider observations, aligned metadata and the recorded audit-workspace layout are still required for numerical replay.

## Public original prediction release

The [v1.1.0-original-predictions Release](https://github.com/yizou728282/ppnn-coldstart-code/releases/tag/v1.1.0-original-predictions) publishes the preserved Stage-2/3/4 and Stage-5 predictive outputs. Copies of the manifests and `SHA256SUMS.txt` are also in `prediction_manifests/`. Every unchanged prediction file retains its original SHA-256. A Stage-2 reference file has its observation field removed while preserving predictive array payloads. Indexes omit observations. Source archives and published shards were checksum-checked, and the packaged files were independently read back and checked against their manifests. These are file-integrity checks, not additional numerical result checks or new experiments.

