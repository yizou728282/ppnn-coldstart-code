# Frozen audit sources

These are the source versions used by the preserved 2026-09-29 numerical audit, from research commit `8858ddbcffb146d82030a2ba126502eb0c4784be`:

- `metrics.py`: `src/ppnn_residual/metrics.py`
- `stage5_inference.py`: `scripts/stage5_inference.py`
- `stage5_supplementary.json`: `protocols/stage5_supplementary.json`

File identities are recorded in `SHA256SUMS.txt` and `provenance/file_origins.json` at the repository root. The audit loaders use these copies before attempting to retrieve files from the historical Git commit.

Executing the full audit still requires provider observations, aligned evaluation metadata and the original audit-workspace layout described in the root `PREDICTIONS.md`. Loading the frozen sources does not perform a numerical replay.
