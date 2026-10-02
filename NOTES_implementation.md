# Implementation notes / deviations (for the methods section)
* The implementation was written from scratch; no earlier local implementation was reused.
* **EMOS parameterization:** σ² = c² + d²·s² instead of c + d·s². This keeps σ² positive without the reference code's penalty, and the two are identical when the optimal c, d ≥ 0.
* **EMOS fitting:**
  * A first version fitted all stations jointly with torch L-BFGS. It had not converged (max |grad| = 0.88) and was replaced by per-station scipy L-BFGS-B with an analytic gradient (unit-tested against finite differences and against a Nelder–Mead reference).
  * Final fit: max |grad| = 1.7e-7 over 532 stations. 4 stations return scipy status "ABNORMAL" (line-search precision) although their gradient is below 1e-8, so they have converged.
  * Stations with fewer than 10 fitting rows are skipped, as in the authors' R code. This affects only station 474.
* **Network σ:** `nn_aux_emb` keeps the paper's linear output with σ = |·|.
* **Station dropout:** 5% of training samples get the `UNK` embedding, so unseen stations have a trained fallback. This differs from the paper, which does not handle unseen stations for the networks, and it is applied equally to all embedding models.
* **Early stopping:** uses the whole 2015 year (not a random 20% subset) to keep years isolated. Refit on 2007–2015 with the selected epoch count.
* **Normalization:** z-score using fitting-period statistics, not the notebook's MAX scaling.
* **Seeds:** 3 per network, not 10.
* **Raw ensemble:** scored as N(ens mean, ens sd), because the dataset contains only the ensemble mean and variance, not the members.
* **Hardware and runtime:** CPU only (8 cores, no GPU). The full pipeline takes about 10 minutes.