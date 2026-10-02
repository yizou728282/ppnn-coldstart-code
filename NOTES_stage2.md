# Stage 2 implementation notes and deviations from the pre-registered protocol

The protocol was committed as `eeb5401`, before any Stage-2 model was trained. Every deviation is listed here.

1. **Boosted EMOS implementation fix (a bug, found using validation data only).**
   * The first `BoostedEMOS` boosted the raw temperature response. The location gradient (y−μ)/σ² is tiny while σ starts at sd(y) ≈ 7 °C, so with the pre-registered 1000 iterations and step 0.05 the fit was far from converged.
   * Validation evidence (fit 2007–2014, score 2015), local model: CRPS 0.921 at 1000 iterations, 0.862 at 5000 and still falling (`results/stage2/logs/bst_iteration_sensitivity_val2015_unstandardized_y.txt`).
   * Fix: standardize y inside the booster and back-transform the predictions. This makes the algorithm invariant to units; the AIC argmin is unchanged apart from a constant.
   * After the fix, with the protocol settings unchanged (1000 iterations, step 0.05, AIC):
     * local validation CRPS: 0.8544 at 1000 vs 0.8537 at 3000
     * global validation CRPS: 0.97582 at 1000 vs 0.97577 at 2000
     * (`results/stage2/logs/bst_iteration_sensitivity_val2015_standardized_y.json`)
   * All EMOS-bst predictions were regenerated. The superseded fit logs are in `results/stage2/superseded_unstandardized_bst/`.
   * Disclosure: before the fix I also saw the 2016 reference score of local EMOS-bst on seen stations (0.8675, against 0.80 in the paper). That reference-reproduction gap was one trigger for the check. After the fix the reference gives 0.8020. No network or proposed-method choice used 2016 data.
2. **Global EMOS-bst AIC stopping always reaches the 1000-iteration cap.** With about 1.2 million rows the AIC penalty is negligible. Validation shows the model has converged by 1000 iterations (point 1).
3. **Timing test runs** used seeds 97–99 (not protocol seeds) on fold 0 and were deleted before the real runs.
4. **Runtime:**
   * About 1.3 h wall clock on 8 CPU cores, as 2 worker processes × 4 threads plus one baselines process.
   * Networks: 455 runs taking 7461 s of summed per-run time; median early-stopping epoch 5 (max 13).
   * Baselines: about 7 min per fold type.
5. **Held-out stations in other years** were not scored. Only 2016 is reported, because held-out stations' 2007–2015 records are excluded from everything by design.
6. **Station attributes** for the attribute MLP are lat, lon, alt, orog and alt − orog, standardized with the training-station statistics of each fold. All networks also receive lat, lon, alt and orog as ordinary input features, as in Rasp & Lerch.
