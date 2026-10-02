# Stage 4, Step 2: why did hybrid_res help on EUPPBench but not on German data?

**EXPLORATORY: not pre-registered.** This is a post-hoc diagnosis that uses only the stored Stage-2 and Stage-3 predictions. No model was trained. Nothing here may be used as a confirmatory claim; it only motivates the pre-registered Stage-4 work.

* Script: `scripts/stage4_diagnose.py`, run time about 10 s.
* Tables: `results/stage4_analysis/` (`swap_decomposition.csv`, `concentration.csv`, `correlations.csv`, `bias_spread.csv`, `per_lead.csv`, `per_station_{stage2,stage3}_{random,spatial}.csv`, `diagnose_stdout.txt`).
* Figures: `figures/stage4_analysis/`.
* All CRPS values are for held-out stations and use the seed ensemble (10 seeds for random folds, 3 for spatial).
* `nn_hybrid_res` differs from `nn_hybrid` in two ways: a residual mean (μ = ensemble mean + f(x)) and a constrained spread (σ² = c² + d²·s²). The embedding (attribute MLP + free offset) and the inputs are identical.

## The discrepancy
| CRPS (°C) | German random | German spatial | EUPPBench random | EUPPBench spatial |
|---|---|---|---|---|
| nn_hybrid | 0.8520 | 0.8960 | 0.9250 | 1.1131 |
| nn_hybrid_res | 0.8509 | 0.8977 | 0.9127 | 0.9803 |
| hybrid − hybrid_res | +0.0011 | −0.0017 | +0.0124 | **+0.1328** |

On German spatial folds hybrid_res is *not* better. It is slightly worse, and the difference is not significant: DM p = 0.31 (`results/stage2/significance_spatial.csv`).

## Findings

1. **The gain comes from the residual mean. The spread head plays no part** (`swap_decomposition.csv`). To separate the two, combine the μ of one model with the σ of the other:

   | EUPPBench spatial | σ_hybrid | σ_res |
   |---|---|---|
   | μ_hybrid | 1.1131 | 1.1100 |
   | μ_res | 0.9816 | 0.9803 |

   * EUPPBench random shows the same pattern: 0.9250 → 0.9132 when only μ is swapped.
   * On German data every combination lies within 0.002.

2. **A handful of stations with out-of-range attributes carries most of the effect** (`concentration.csv`, `figures/stage4_analysis/concentration_curves.png`, `station_diff_vs_novelty_distance.png`).
   * EUPPBench spatial: the top 5 of 117 stations produce 75% of the pooled difference. Top-5 contributions are +0.35 for Galtür, +0.14 for Warth, +0.11, +0.11 and +0.03, out of +0.133.
   * These stations are:
     * the Alpine valley stations Galtür (AT, 1587 m) and Warth (AT, 1478 m);
     * DE 303730 (806 m);
     * FR 207460 (376 km from the nearest training station);
     * Feldkirch (AT).

     The four Austrian stations all sit in spatial fold 4, and FR 207460 sits in fold 1.
   * Station-mean CRPS at Galtür:

     | method | CRPS |
     |---|---|
     | raw ensemble | 4.77 |
     | emos_bst_gl | 1.65 |
     | nn_unk | 3.41 |
     | nn_noemb | 4.15 |
     | nn_attr | 8.30 |
     | nn_hybrid | 7.01 |
     | nn_hybrid_res | 1.55 |

   * What goes wrong at Galtür: the raw ensemble is 5.4 °C too cold there, because the model orography of 2105 m is 518 m above the station. The non-residual networks overcorrect, and nn_hybrid ends up with a +8.7 °C station bias. nn_hybrid_res ends at +0.05 °C.
   * Split by whether any attribute (lat, lon, alt, orog, alt−orog) lies outside the fold's training-station range:
     * 21 such stations: pooled Δ = +0.576
     * the other 96 stations: Δ = +0.035
   * Excluding the top 5 stations, hybrid_res is still better (+0.034), and 80% of stations have a lower CRPS with it. So on EUPPBench spatial folds there is also a smaller, broad effect. Stations inside a fold are spatially correlated, so this fraction is not an independent-sample test.
   * The per-station Δ also rank-correlates with attribute novelty: ρ = 0.26, p = 0.004, where novelty is the distance in standardized attribute space to the nearest training station.

3. **Attribute extrapolation is not bad in general. The sign and size depend on *which* direction is extrapolated.**
   * The German data contain a more extreme extrapolation: Zugspitze, 2964 m, 1474 m above the highest training station in its spatial fold, with attribute novelty 9.3. There, hybrid and hybrid_res are equally good (1.19 vs 1.15; station bias −0.03 vs −0.24 °C), and the raw ensemble is 5.3 °C too warm.
   * The two cases differ in direction:
     * Zugspitze is a **summit** (alt − orog = +1696 m). German training folds contain many stations far above the model orography (alt−orog up to +954 m), so the correction extrapolates in a direction the training data span.
     * Galtür, Warth and Feldkirch are **valley** stations far *below* the model orography (alt−orog = −518, −274 and −534 m; corrected 2026-09-26 after the independent number audit, earlier text said −534 and −275 for Galtür and Warth). The training stations of that fold reach at most −191 m, and the valley stations also lie at the southern and eastern edge of the domain.
   * An unconstrained MLP output extrapolates this almost linearly, with an implausible slope: a +14 °C correction at Galtür.
   * The residual parameterisation keeps μ tied to the ensemble mean. It limited the damage in this case, but this is one mechanism observed on a few stations, not a general guarantee.
   * On German spatial folds, hybrid_res is slightly *worse* on the 29 out-of-range stations (Δ = −0.051). These are mostly mid-altitude hill stations where both models share a warm bias.

4. **The attribute MLP amplifies the failure but does not cause it by itself.**
   * nn_noemb has no attribute MLP; it takes the attributes only as plain inputs. It also fails on EUPPBench spatial (1.076; Galtür 4.15).
   * nn_attr, whose embedding comes only from attributes, fails worst (1.156; Galtür 8.30).
   * nn_unk (0.983) is only mildly affected. It gets the same plain attribute inputs, but its free embedding absorbs station-specific offsets during training, so the weights on the attribute inputs probably stay smaller. This is a hypothesis and has not been tested.
   * hybrid_res uses the same attribute MLP as hybrid, so the difference cannot come from the embedding.

5. **Lead time is not the explanation** (`per_lead.csv`, `figures/stage4_analysis/stage3_diff_by_lead.png`).
   * On EUPPBench spatial the hybrid − hybrid_res gap is roughly constant across leads: +0.137, +0.140, +0.135, +0.133 and +0.118 at 24, 48, 72, 96 and 120 h.
   * At 48 h, the only lead in the German data, it is +0.140 on EUPPBench against −0.002 on German data.

6. **Station density matters only through extrapolation.**
   * In the EUPPBench spatial folds, 96 of 117 held-out stations have no training station within 50 km.
     * For these isolated stations Δ = +0.156; for the others Δ = +0.026.
   * In the German spatial folds, 340 of 502 held-out stations are also isolated, and there Δ = −0.004.
   * Being isolated is therefore not enough. What matters is that the sparse EUPPBench network (117 stations, 5 countries, 4 Alpine stations all in one fold) leaves parts of the attribute space empty, while the 535 German stations cover it.
   * Distance alone correlates only weakly with Δ: EUPPBench spatial ρ = 0.20, p = 0.03; German spatial ρ = −0.13.

7. **Calibration** (`bias_spread.csv`, `figures/stage4_analysis/pit_hybrid_vs_res.png`).
   * On EUPPBench spatial, hybrid has:
     * a large mean |station bias| (0.84 vs 0.59 for hybrid_res and 0.43 for emos_bst_gl)
     * too-narrow effective intervals (cov80 0.739 vs 0.781)
     * spread–skill 0.79 vs 0.95
   * σ is almost the same for the two models (1.61 vs 1.64). The under-dispersion comes from the μ errors, not from σ.
   * On German data the two models are indistinguishable (cov80 0.743 vs 0.745).

## Diagnosis in plain words
* On EUPPBench, hybrid_res beats hybrid mainly because a few held-out stations, mostly Alpine valley stations and one far-away French station, have station attributes outside what the training stations cover.
* At those stations the ordinary networks extrapolate the "station characteristics → temperature correction" relation linearly and produce forecasts that are several degrees wrong.
* Tying the predicted mean to the raw ensemble mean (residual output) limited this error. The spread formulation plays no role.
* The German network is dense and covers the attribute space well, and its most extreme station (Zugspitze) is extrapolated in a physically "easy" direction, so the two parameterisations perform the same there.
* So the residual head is not a generally better model. It is a safeguard against extrapolation, and it pays off only when held-out stations fall outside the training attribute range.

## Consequences for Stage 4 (design input, not evidence)
* Any new method should avoid unconstrained extrapolation in attribute space. Two ways:
  * standardise target and forecasts climatologically (SAMOS-style), so the network models anomalies;
  * use the forecasts of nearby training stations (GNN or attention) instead of extrapolating from attributes.
* The spatial folds, and especially EUPPBench folds 1 and 4, are the relevant stress test. Stage-4 reports should give per-fold and out-of-range-station results as secondary diagnostics.
* A residual (ensemble-anchored) mean should be the default output parameterisation for new networks. This choice is made *now*, before any Stage-4 run, based on this exploratory analysis.
