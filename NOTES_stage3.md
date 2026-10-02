# Stage 3 implementation notes (EUPPBench validation)

The protocol is `protocols/stage3_euppbench.{md,json}`, committed as f0f6517 before any Stage-3 model run.

## Data acquisition
* **Source:** Zenodo record 7708362, "EUPPBench postprocessing benchmark dataset - station data", v1.0 (2023-03-08).
  * Files: `EUPPBench-stations.zip`, 18,324,361,137 bytes, md5 e409457279b3494d18f2dfb41f3f449b; and `LICENSE`, md5 b82e16afa33c93fdc5fc5a6af62eb3e3, copied to `results/stage3/EUPPBench_LICENSE.txt`.
  * Licence id "other-at". ECMWF forecasts: CC BY 4.0. Observations: RMIB, Météo-France, ZAMG, KNMI, DWD. Land use and altitude: Copernicus CLMS/EEA.
* **Why not the full zip:** a direct download from Zenodo ran at about 0.7 MB/s (roughly 7 h), so it was stopped after 67 MB.
* **What was done instead:**
  * `scripts/stage3_download.py` reads the zip's central directory via HTTP range requests. The zip is uncompressed (STORED; 139,764 members, each with a CRC32).
  * It fetches only the needed members from the official climetlab/ECMWF object store (`object-store.os-api.cci1.ecmwf.int/eumetnet-postprocessing-benchmark-1st-phase-training-dataset`).
  * Every file is verified by size and CRC32 against the Zenodo directory: 45,346 files, 9.43 GB, 0 mismatches, about 25 min.
  * Manifest: `results/stage3/stage3_download_manifest.json`.
  * **Caveat:** the md5 of the whole 18.3 GB zip was not checked, because the whole zip was never downloaded. Integrity rests on the per-member CRC32 values.
* **Stores used, for each of AT, BE, FR, DE, NL:**
  * ensemble {reforecasts, forecasts}: surface, surface_postprocessed, pressure_850
  * {reforecasts, forecasts}: observations_surface
* **Dataset layout:**
  * Reforecasts: [station, model date (209 Mondays/Thursdays of 2017–2018), member 11, year 20, step 21 (0–120 h by 6 h)].
  * The "year" index k = 1..20 means the same calendar day k … 20 years back: 1997–2016 for 2017 model dates, 1998–2017 for 2018 model dates.
  * Forecasts: 730 daily 00 UTC runs (2017-01-01 … 2018-12-31), 51 members.
* **Stations:**
  * 122 stations in total: AT 4, BE 29, FR 8, DE 51, NL 30. No Swiss stations; their data are not public.
  * Altitude −4 to 1587 m.
  * **117 stations are included** under the pre-registered ≥ 50% observation rule. Excluded:
    * BE 106439 Sint-Katelijne-Waver: 47% of reforecast-period observations present
    * BE 106438 Stabroek: 23%
    * FR 207072 Reims-Prunay: 24%
    * DE 303362 Mühlacker: no observations
    * DE 305347 Warburg: no observations
* **Preprocessing** (`scripts/stage3_preprocess.py`, 40 s):
  * Output: `data/stage3/stage3_euppbench.npz`, 361,868,940 bytes, sha256 0ade3652…d5d1.
  * 2,296,571 training rows (reforecasts valid 1997-01-03 … 2016-12-31) and 423,090 test rows.
  * 65,026 reforecast rows valid on or after 2017-01-01 were dropped (overlap with the test period).
  * Rows dropped for a missing observation: 169,192 in training and 11,590 in test.
  * No other predictor has missing values.
  * Per-station table: `results/stage3/stations.csv`. Counts: `results/stage3/stage3_preprocess_info.json`.

## Code changes made before any Stage-3 model run
* **GPU support:**
  * New `src/ppnn_residual/device.py`: device auto-detection, seeding, deterministic flags, device info written into every run log.
  * `StationNet` now moves its dropout mask, drawn on the CPU generator, to the input's device. This is a no-op on CPU.
  * Stage-3 training draws permutations and dropout masks from seeded CPU generators, so the random stream is device-independent.
  * CPU reproducibility check: Stage-2 hybrid, random fold 0, seed 0, 4 threads, re-run after the changes. The result is **bitwise identical** to the stored Stage-2 prediction.
* **Boosting:** `BoostedEMOS` now never selects a zero-variance column.
  * Without this, a land-use class missing from a fold's training stations gave 0/0 = NaN.
  * The behaviour is identical when no such column exists, as in Stage 2 (all Stage-2 fits were finite).
* **Tests:** `tests/test_stage3.py` covers device detection, bitwise CPU reproducibility for 3 variants, GPU vs CPU agreement and GPU reproducibility (skipped without CUDA), ensemble statistics, the land-use mapping, the resumable seen-station accumulator and the boosting constant-column fix. Result: 23 passed, 1 skipped (CUDA).

## Smoke test (box, CPU)
* The full Colab runner (`colab/stage3_colab.py --smoke --device cpu`) ran end to end: 30-station subsample, 1 seed, 1 epoch, 50 boosting iterations. It covered baselines, reference, both fold types, evaluation and both zips. It took 2.8 min.
* A re-run skipped all finished work (0.01 h).
* Its outputs were deleted and are not reported.
* Timing:
  * Full-size CPU epoch on the box (8 threads, 1.76 M rows): 1.8 s (unk) to 2.9 s (hybrid).
  * One full-size boosted EMOS fit (320k × 44, 1000 iterations, 1 thread): 22 s.

## Change of compute plan (2026-09-26, deviation from the protocol's "compute" section)
* **Why:** Colab quota was limited. The Colab T4 run was stopped after about 40 min, and Stage 3 now runs on the local 8-core CPU machine.
* **Partial Colab results** (`results_partial.zip`, 228 MB, `unzip -t` OK). Contents:
  * 61 finished T4 network runs on random folds: fold 0 with all 5 variants × 10 seeds; fold 1 `unk` seeds 0–9 and `noemb` seed 0.
  * Random-fold baselines for all 7 folds; spatial-fold baselines had not started.
  * Every T4 log had its prediction files, and every seen-station accumulator listed exactly the logged seeds. There were no orphan or temporary files, so nothing was half-written.
* **CPU and T4 are not bit-identical.** Rerunning `unk`, fold 0, seed 0 on CPU (4 threads):
  * Early-stopping curves agree at first and then drift. Best epoch 12 in both; best validation CRPS 0.91797 (CPU) vs 0.91810 (T4).
  * Held-out predictions differ by mean |Δμ| = 0.079 °C and max 0.66 °C.
  * *Added 2026-09-26 (number audit):* this rerun's predictions were not saved, so 0.079 °C cannot be recomputed. The saved final CPU predictions differ from the T4 predictions by mean |Δμ| = 0.176 °C for unk/fold 0/seed 0 and 0.117 °C averaged over all 61 T4 runs (max 3.4 °C; `provenance/derived_numbers.json`, key `cpu_vs_t4_gpu_final_predictions`). CPU runs are therefore not bit-identical across reruns either (e.g. different thread counts).
  * The same seed on two devices is therefore a different network of equal quality, not the same network.
  * The earlier bitwise check was CPU vs CPU (the Stage-2 rerun). The GPU test never ran on Colab: the test module failed to import because `zarr` was not installed there. That is now fixed by importing zarr lazily.
* **Decision: all Stage-3 networks are trained in ONE environment, the local CPU**, so that no seed ensemble mixes devices (the spirit of the protocol's single-environment rule).
  * The 61 T4 runs are kept apart in `results/stage3/colab_t4_partial/` (logs and colab_logs committed; predictions git-ignored). They are not used in the main results, only as an optional device-sensitivity comparison.
  * Every run log records its device (`device`, `threads`, `torch`).
* **Baselines:** the random-fold baselines from the Colab CPU are reused, since they are deterministic numpy code. The spatial-fold baselines and the local-bst reference are computed on the box.
* **Launch:** `scripts/stage3_cpu_launch.sh`. Two lanes:
  * random networks with 5 threads;
  * spatial baselines, then the reference (3 BLAS threads), then spatial networks (3 threads).
  * Evaluation runs at the end.

## Run record (local CPU)
* `scripts/stage3_cpu_launch.sh`, started 2026-09-26 09:03 and finished 11:33 (UTC-6; the launcher log stores UTC).
  * Spatial baselines + local-bst reference: 13 min.
  * Spatial networks: 105 runs, 3,439 s summed, 3 threads.
  * Random networks: 350 runs, 8,923 s summed, 5 threads.
  * Median best epoch 9 (max 18). All logs record `device=cpu`.
* **Colab T4 runs discarded.** The 61 T4 network runs (random fold 0 × 5 variants × 10 seeds; fold 1 `unk` s0–9 and `noemb` s0) were **not** used. All 455 networks in the main results were retrained on the box CPU, so no seed ensemble mixes devices. The T4 logs remain in `results/stage3/colab_t4_partial/` for an optional device-sensitivity comparison.
* **Random-fold baselines** are the deterministic numpy fits from the Colab CPU (see above).
* **Evaluation:** `scripts/stage3_evaluate.py --out results/stage3 --fig figures/stage3`. All tables below are copied from `results/stage3/summary_*.csv`, `significance_*.csv`, `per_lead_*.csv` and `distance_bins_*.csv`.
* **Figures:** `figures/stage3/{crps_vs_distance,pit_heldout,map}_{random,spatial}.png`.

## Results: held-out stations, test 2017-01-01 … 2018-12-31, leads 24–120 h (423,090 rows)
Seed-ensemble CRPS (°C). Random folds use 10 seeds; spatial folds use 3 seeds. "Seen" is the CRPS of the same models on their own training stations over the test period (mean over folds).

| method | CRPS random | CRPS spatial | seen (random) | cov80 random / spatial | PIT RI random / spatial | spread-skill random / spatial |
|---|---|---|---|---|---|---|
| raw_ensemble_gauss | 1.2113 | 1.2113 | 1.2113 | 0.512 / 0.512 | 0.576 / 0.576 | 0.58 / 0.58 |
| emos_gl | 1.1139 | 1.1395 | 1.1104 | 0.817 / 0.811 | 0.093 / 0.095 | 0.98 / 0.97 |
| emos_bst_gl | 0.9625 | 0.9770 | 0.9543 | 0.844 / 0.838 | 0.111 / 0.108 | 1.07 / 1.19 |
| nn_unk | **0.9122** | 0.9832 | 0.8218 | 0.799 / 0.777 | 0.065 / 0.069 | 0.96 / 0.94 |
| nn_knn | 0.9190 | 0.9801 | 0.8218 | 0.797 / 0.778 | 0.113 / 0.088 | 0.97 / 0.94 |
| nn_noemb | 0.9385 | 1.0759 | 0.8351 | 0.789 / 0.745 | 0.062 / 0.126 | 0.95 / 0.89 |
| nn_attr | 0.9368 | 1.1560 | 0.8285 | 0.788 / 0.729 | 0.062 / 0.146 | 0.93 / 0.76 |
| nn_hybrid (Stage-2 proposal) | 0.9250 | 1.1131 | 0.8259 | 0.795 / 0.739 | 0.057 / 0.131 | 0.95 / 0.79 |
| nn_hybrid_res (exploratory) | 0.9127 | 0.9803 | 0.8229 | 0.798 / 0.781 | 0.057 / 0.083 | 0.96 / 0.95 |
| nn_hybrid_knn (exploratory) | 0.9255 | 1.1119 | 0.8259 | 0.795 / 0.739 | 0.057 / 0.131 | 0.95 / 0.79 |

* On spatial folds, **emos_bst_gl (0.977) has the lowest CRPS of all methods**. It is followed by nn_knn at 0.980 and nn_hybrid_res at 0.980.
* **Single-seed CRPS (mean ± sd), random folds:**
  * nn_unk 0.9315 ± 0.0055
  * nn_hybrid 0.9505 ± 0.0069
  * nn_hybrid_res 0.9354 ± 0.0110
* **Reference, seen stations:** local boosted EMOS per station and lead, fit on 1997–2016 reforecasts, gives CRPS 0.8523. The networks on their own training stations reach 0.82–0.84.
* **11-member sensitivity:** all methods get worse by 0.007–0.010. The ranking does not change (`sensitivity_11members_*.csv`).
* **CRPS per lead, random folds:** the ordering is the same at every lead. nn_unk and nn_hybrid_res are best, from 0.749 at 24 h to 1.151 at 120 h.
* **Spatial folds:** most of the damage to noemb, attr and hybrid comes from **spatial folds 1 and 4**. Fold 1 is 3 French stations. Fold 4 is 17 stations: the 4 Austrian stations, 2 French and 11 German, with altitudes up to 1587 m. For example, nn_hybrid scores 2.256 on fold 1 and 1.779 on fold 4, against 1.243 and 1.169 for emos_bst_gl (`per_fold_spatial.csv`). See `NOTES_stage4_analysis.md`.

### Primary significance tests (nn_hybrid vs X; DM on 730 daily means, NW lag 5, BH q = 0.05, 7-day block bootstrap)
| X | random: Δ (rel.) , 95% CI | spatial: Δ (rel.) , 95% CI | claim rule |
|---|---|---|---|
| emos_gl | −0.189 (−17.0%), [−0.203, −0.173] | −0.026 (−2.3%), [−0.047, −0.002] | **hybrid better** |
| emos_bst_gl | −0.038 (−3.9%), [−0.046, −0.029] | +0.136 (+13.9%), [+0.124, +0.150] | no claim (sign flips) |
| nn_unk | +0.013 (+1.4%), [+0.011, +0.015] | +0.130 (+13.2%), [+0.120, +0.141] | **hybrid worse** |
| nn_knn | +0.006 (+0.7%), [+0.002, +0.010] | +0.133 (+13.6%), [+0.123, +0.144] | **hybrid worse** |
| nn_noemb | −0.014 (−1.4%), [−0.015, −0.011] | +0.037 (+3.5%), [+0.031, +0.044] | no claim (sign flips) |
| nn_attr | −0.012 (−1.3%), [−0.014, −0.010] | −0.043 (−3.7%), [−0.049, −0.038] | **hybrid better** |

All 6 random-fold primary tests are BH-significant (`results/stage3/hypotheses_and_claims.json`).

### Replication hypotheses carried over from Stage 2 (pre-registered)
| | statement | outcome | evidence |
|---|---|---|---|
| H1 | every network beats emos_gl and emos_bst_gl on both fold types | **not supported** | spatial folds: every network is worse than emos_bst_gl (0.977); nn_attr is also worse than emos_gl |
| H2 | random folds: hybrid, attr and noemb within 1% of each other, all beating unk | **not supported** | the spread is 1.5%, and nn_unk (0.912) beats all three |
| H3 | spatial folds: noemb beats attr and hybrid | **supported** | 1.076 < 1.113 and < 1.156 |
| H4 | knn worse than unk on both fold types | **not supported** | worse on random (0.919 vs 0.912) but better on spatial (0.980 vs 0.983) |
| H5 | seen stations: hybrid beats noemb | **supported** | 0.826 < 0.835 |

### What this means for the Stage-2 proposal
* On EUPPBench the pre-registered proposal nn_hybrid does **not** replicate as the best new-station method.
* On random folds it is significantly worse than nn_unk and nn_knn.
* On spatial folds it is far worse than emos_bst_gl, nn_unk, nn_knn and nn_hybrid_res. The attribute MLP extrapolates badly to held-out regions whose altitude and latitude lie outside the training range.
* The exploratory nn_hybrid_res is statistically tied with the best method on both fold types. It stays exploratory under the Stage-2/3 rules, and any follow-up must be pre-registered as new work (Stage 4).
