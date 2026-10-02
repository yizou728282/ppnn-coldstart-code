# Transfer of neural ensemble temperature post-processing to unobserved stations

Code, protocols, evaluation summaries and preserved-prediction manifests for a station-held-out evaluation with random and spatially blocked folds.

- **Author:** Yi Zou, independent researcher, Xi'an, China.
- **Research source:** original development commit `7213a55cfef8b195d8d9b283941933b091553130`.
- **This repository:** one snapshot of the implementation and saved evaluation outputs. The manuscript and the journal submission package are not included.
- **Publication status:** author review pending; not submitted, accepted or published by a journal.
- **AI use disclosure:** GPT-5.6 (OpenAI) was used solely for language polishing of the text; the author retains full responsibility for the manuscript.

The main result concerns evaluation design. On EUPPBench, the DRN and geographic GNN deteriorate relative to boosted EMOS under spatial blocking, with positive contrasts across six additional partitions. **A robust ranking reversal is not established.** The compared pipelines differ in predictors and validation details; the conclusions do not isolate an architecture or station-density effect.

## Contents

| Directory | Contents |
|---|---|
| `src/`, `scripts/` | Research implementation, data download and preprocessing, training, evaluation and audit helpers |
| `configs/`, `protocols/` | Configurations and copies of the internally recorded protocols |
| `results/` | Aggregate evaluation summaries and statistical comparison tables |
| `analysis/` | Matched-seed, forecast-only and regional-block DiD calculations |
| `handoff/original_prediction_audit/` | Audit code, field checks and recomputed aggregate results |
| `NOTES_*.md` | Implementation and experiment records |
| `prediction_manifests/` | Inventories and SHA-256 checksums of the preserved prediction archives |
| `provenance/` | Saved numerical reference values (`derived_numbers.json`, `stage5_numbers.json`) |
| `figure_data/` | Primary random and spatial fold assignments |

## Interpretation of protocol copies

The analysis definitions, dates and decision rules in `protocols/` are retained. Historical words “registered” and “pre-registered” refer to internally recorded working plans, not registration with an external registry or an independently verifiable public preregistration.

## Research data and experimental reproduction

Original data are obtained directly from their providers:

- German station data: [figshare, DOI 10.6084/m9.figshare.13516301.v1](https://doi.org/10.6084/m9.figshare.13516301.v1), CC BY 4.0.
- EUPPBench: [Zenodo, DOI 10.5281/zenodo.7708362](https://doi.org/10.5281/zenodo.7708362). Forecasts are CC BY 4.0; use the observation licence included with that record.

Preserved per-seed predictions are in the [v1.1.0-original-predictions Release](https://github.com/yizou728282/ppnn-coldstart-code/releases/tag/v1.1.0-original-predictions). See [PREDICTIONS.md](PREDICTIONS.md) for verification, restoration and alignment. Raw observations, processed observation metadata with observation values and model checkpoints are not redistributed. The code licence does not override dataset licences.

```bash
python -m venv .venv
# Activate the environment using the command appropriate to your operating system.
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pytest -q tests
python scripts/download_data.py
python scripts/stage3_download.py --out data/euppbench/zarr
python scripts/stage3_preprocess.py --zarr data/euppbench/zarr --out data/stage3/stage3_euppbench.npz
```

Follow the stage-specific protocols and `NOTES_stage2.md` through `NOTES_stage5.md` for the recorded sequence of training, evaluation and supplementary inference. Shell launchers retain the original interpreter default: set `PY=python` when using a different environment. The reported runs used CPUs; exact bitwise equality on other hardware or thread settings is not promised.

Regenerating manuscript files and recomputing statistics from per-seed predictions are different operations. `scripts/make_paper_figures.py` loads predictions and cannot be run as a summaries-only replay. Independent recomputation still requires provider observations, exact row alignment, the appropriate fold assignments and preparation of the original evaluation workspace. Do not substitute incomplete reruns for the reported ensemble design.

## Verification boundary

The original 2026-09-29 audit reports 1,986 passing score and test field checks and 112 passing core station and seed difference-in-differences checks. Those reports were inspected; training and the full numerical audit were not rerun for this snapshot. Not every auxiliary ranking, fold-block analysis, sensitivity or figure was regenerated. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Licence and archival status

The research implementation is MIT-licensed; see [LICENSE](LICENSE). Dataset terms remain those of the data providers. No archival DOI is claimed. A GitHub repository is not a Zenodo deposit. See [archive/ARCHIVING.md](archive/ARCHIVING.md).
