# Stage 5 supplementary experiments: delivered results and audit

Source: GitHub commit `8a7c56d` (results committed in `d2b2d43`), downloaded
2026-09-28. Protocol: `protocols/stage5_supplementary.md` and `.json`.
Final reports: `results/stage5/eval/final/`.

**2026-09-29 recovery update:** `original-preds-stage2-4` now supplies the original
predictions. The audit in `handoff/original_prediction_audit/` reproduces 70 original
score combinations, 124 daily comparisons, and the core DRN/GNN station/seed DiD
with all six additional partitions. All 1,986 score/test and 112 core-inference
field checks pass. This supersedes the missing-original-input status of the
2026-09-28 audit described below; it does not claim every auxiliary analysis was rerun.

## Completion and provenance

The original Linux run started at 2026-09-27 05:22:23 UTC and completed at
21:04:52 UTC (15 h 42 min; 2026-09-27 15:04 UTC-6). All 47 launched jobs have
an exit-0 completion record. No scope cut occurred. The final inference job
also completed successfully. These are the original experiment results; the
partial Windows reconstruction is not used to replace or combine with them.

The delivery audit checked 1,957 expected training-log files, complete
10-seed spatial ensembles for all 11 network outputs on both datasets, and
all six additional EUPPBench partitions. The saved C1-C3 verdicts agree with
their registered conditions. See `handoff/STAGE5_DELIVERY_AUDIT.json` and
`scripts/audit_stage5_delivery.py`.

The release `stage5-preds-2026-09-27` contains 2,519 Stage-5 prediction files,
3,525,515,286 extracted bytes. Both archive-part SHA-256 values match the
digests returned by GitHub. The local manifest is
`local_runs/release_download/restored.json`. Original Stage-2/3/4 predictions
are outside this release. The final CSV/JSON reports already include those
original predictions, but the entire inference pipeline has not been
recalculated locally from all original inputs. The separate prediction audit
recalculates the available Stage-5 scores without overwriting final reports;
see `scripts/audit_stage5_predictions.py` and its JSON output in `handoff/`.
All 241 recalculated scores agree within the five-decimal CSV rounding
tolerance (maximum absolute discrepancy 0.000004972 degrees C). The partial
Windows predictions and their completion markers have been moved to
`local_runs/windows_prediction_archive/` so a later inference call cannot
accidentally mix them with the original Linux seed ensembles.

## Registered conclusions

Let E be global Boosted EMOS and let DiD = (F-E)_spatial - (F-E)_random.
Positive values mean that F loses relative skill under spatial blocking.
Spatial ensembles use 10 seeds. The DRN/GNN random ensembles retain the
registered three seeds, so ensemble sizes differ in this comparison.

| Dataset | F | DiD (degrees C) | Seed-nested station 95% CI |
|---|---|---:|---|
| EUPPBench | DRN | 0.15057 | [0.04114, 0.31737] |
| EUPPBench | GNN-Geo | 0.10224 | [0.03725, 0.18274] |
| German | DRN | 0.00815 | [-0.00100, 0.01211] |
| German | GNN-Geo | 0.00617 | [-0.00895, 0.01451] |

- **C1 supported:** both EUPPBench intervals exclude zero and both methods
  have a positive DiD in all six additional partitions (rule requires at
  least four). This supports relative deterioration under spatial blocking.
- **C2 not established for either method:** for DRN, the spatial F-E
  station interval is [-0.00553, 0.27469]; for GNN-Geo, the random interval
  is [-0.03322, 0.02051]. A statistically established ranking reversal
  must therefore not be claimed.
- **C3 condition met:** both German DiD intervals include zero. With one
  dataset of each type, density remains confounded with region, forecast
  leads and predictors; this is not causal evidence about density.

Regional concentration remains important. The EUPPBench spatial fold-block
F-E intervals include zero for DRN and GNN-Geo. In the final ten-seed
per-fold table, DRN deficits are especially large in paper fold 2 (three
French stations) and fold 5 (17 stations including Austria). GNN-Geo's
largest deficit is in fold 5. C1 wording must retain the regional caveat.

## Ten-seed rankings and new statistical baselines

EUPPBench spatial CRPS: NN-Hybrid-Res 0.97435, NN-UNK 0.97599,
Boosted EMOS 0.97700, NN-kNN 0.98036, SAMOS-Attn 0.98807,
SAMOS-MLP 0.99442, GNN-Geo 1.06351 and DRN 1.08207 degrees C.
Thus the old three-seed statement that Boosted EMOS has the *lowest pooled
score* must not be applied to the ten-seed results. Its differences from
NN-UNK and NN-Hybrid-Res are not statistically established under C4/C5.
DRN is also statistically tied with Boosted EMOS under C4 because its
station interval includes zero. GNN-Geo is worse on average over stations,
with the required regional caveat.

On German spatial folds, DRN scores 0.87588 and GNN-Geo 0.89340 versus
Boosted EMOS 0.91578. Both satisfy C4 for improvement, including fold-block
intervals below zero.

The new EMOS trend-plus-IDW baseline scores 1.01775 (EUPPBench spatial)
and 0.96455 (German spatial); linear SAMOS scores 1.01648 and 0.95787.
Both are worse than Boosted EMOS under C4 on both spatial designs. Neither
satisfies C6's condition of beating every EUPPBench network.

## Sensitivity results

**Boosting iterations.** On EUPPBench spatial folds, 1000 iterations score
0.97700 and validation-selected stopping scores 1.00148. Although the
fixed 500-iteration test score is 0.97500, test performance must not be used
to choose the stopping rule. On random folds the corresponding scores are
0.96250 and 0.96110. The registered primary setting remains unchanged.

**Forecast-only neighbour availability (m1a), seeds 0-2.**

| Method/design | Original CRPS | Fixed CRPS | Fixed-original | Station/seed 95% CI |
|---|---:|---:|---:|---|
| GNN-Geo, EUPPBench spatial | 1.04234 | 1.02829 | -0.01405 | [-0.08303, 0.06239] |
| SAMOS-Attn, EUPPBench spatial | 1.02185 | 1.03568 | 0.01383 | [-0.18031, 0.23154] |
| GNN-Geo, EUPPBench random | 0.94678 | 0.94328 | -0.00350 | [-0.02679, 0.01525] |

The spatial changes exceed the registered absolute 0.005-degree reporting
threshold, so corrected and original values must appear together in the
paper. Their uncertainty intervals include zero. Original predictions stay
primary under the protocol. The delivered metadata report 86,368 training
and 3,960 test forecast rows with missing observations. Those counts are
not the fraction of test cases whose neighbour set changes. The latter diagnostic
was recovered from GNN run logs: 10,910 / 423,090 = 2.57865% for both fold types.
For spatial SAMOS-Attn it is 27,370 / 423,090 = 6.46907%, reconstructed from
committed test availability and the eight-nearest-training-station rule.
The reconstruction reproduces the original GNN log counts exactly. See
`scripts/audit_stage5_neighbours.py` and `handoff/STAGE5_NEIGHBOUR_AUDIT.json`.
These are diagnostics of test neighbour selection, not causal attributions
of score changes: the reruns also change availability during fitting.

## Manuscript integration (2026-09-28)

The nine supplementary placeholders have been resolved. Abstract, methods,
design, results, discussion and conclusions now use C1–C6 consistently.
The original three-seed tables and exploratory diagnostics remain labelled;
the final DiD table and Appendix D report the ten-seed spatial scores,
six extra partitions, stopping sensitivity and forecast-only availability.
The text reports station and regional uncertainty separately and does not
claim a robust ranking reversal. The source also explains that Stage 5
cannot retroactively pre-register the earlier narrative.

`scripts/make_stage5_paper_tables.py` writes five tables and
`provenance/stage5_numbers.json` from delivered reports and the neighbour
audit. All new methodological references have DOI metadata. Funding is
confirmed as no specific external funding, with thanks only to the data
providers. The updated PDF has been compiled; visual and text checks are
recorded in `handoff/PAPER_STAGE5_QA.md`.

Remaining submission tasks are journal choice/template/length checks and
an authorized public repository release with an archival DOI. No new
training is required by the completed Stage-5 protocol.
