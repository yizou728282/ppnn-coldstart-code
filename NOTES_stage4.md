# Stage 4: baselines + SAMOS+neighbour-attention pilot

Status as of 2026-09-26 (UTC-6). All numbers from `results/stage4/*/summary_*.csv`, `results/stage4/pilot_decision.json`, and `results/stage4/logs_launcher/final_eval.txt`. Protocols: `protocols/stage4_baselines.*`, `protocols/stage4_pilot.*`.

## Baselines (drn_lak, gnn_geo)

Lakatos-style DRN (`drn_lak`) and distance-graph GNN (`gnn_geo`), 3 seeds, same folds as Stage 2/3.

| CRPS | German random | German spatial | EUPPBench random | EUPPBench spatial |
|---|---|---|---|---|
| drn_lak | 0.8645 | 0.8795 | 0.9170 | 1.0787 |
| gnn_geo | 0.8840 | 0.9048 | 0.9468 | 1.0423 |
| nn_unk | 0.8706 | 0.8922 | 0.9205 | 0.9832 |
| emos_bst_gl | 0.9125 | 0.9158 | 0.9625 | 0.9770 |
| nn_hybrid_res | 0.8531 | 0.8977 | 0.9134 | 0.9803 |

- On **random** folds both baselines beat global boosted EMOS; DRN is competitive with `nn_unk` / slightly better on EUPPBench random.
- On **sparse EUPPBench spatial** folds both lose to `emos_bst_gl` (0.977) and to residual/unk nets (~0.98). DRN 1.079 and GNN 1.042 are clearly worse.
- Pattern matches the Stage-4 exploratory diagnosis: unconstrained attribute-driven nets fail when held-out stations leave the training attribute range.

## Pilot outcome (pre-registered go/no-go)

Method: climate standardization (SAMOS-style) + optional neighbour attention (`samos_attn`); ablation without neighbours (`samos_mlp`). Primary comparators: `nn_unk`, `nn_knn`, `emos_bst_gl`. GO needs BH-significant spatial improvement vs best primary, CI excluding 0, and same-sign random improvement. Otherwise **NO-GO**.

### Phase 1 — German → **NO-GO**

| | best primary | samos_attn | Δ | DM p | BH |
|---|---|---|---|---|---|
| random | nn_unk 0.8706 | 0.8738 | +0.0032 | 0.24 | no |
| spatial | nn_unk 0.8922 | 0.8865 | −0.0057 | 0.10 | no |

Spatial point estimate was slightly better (~0.6%) but not significant (p=0.10, CI crosses 0). Protocol Phase-2 gate was only "promising point estimate on German spatial", so Phase 2 still ran as a replication check.

### Phase 2 — EUPPBench → **NO-GO**

| | best primary | samos_attn | Δ | DM p | BH |
|---|---|---|---|---|---|
| random | nn_unk 0.9205 | 0.9364 | +0.0159 | 1.1e-7 | yes (worse) |
| spatial | emos_bst_gl 0.9770 | 1.0218 | +0.0448 | 1.2e-8 | yes (worse) |

Replication fails: `samos_attn` is significantly **worse** than the best primary on both fold types. `promising_point_estimate_spatial` = false.

## samos_mlp vs samos_attn

| CRPS | German random | German spatial | EUPPBench random | EUPPBench spatial |
|---|---|---|---|---|
| samos_mlp | 0.8544 | 0.8784 | 0.9198 | 1.0023 |
| samos_attn | 0.8738 | 0.8865 | 0.9364 | 1.0218 |

Plain climate standardization (`samos_mlp`) beats the attention version on every split. On German data `samos_mlp` is close to the best nets; on EUPPBench spatial it still loses to `emos_bst_gl` / `nn_hybrid_res` / `nn_unk`. Neighbour attention did not help and often hurt.

## Out-of-range station diagnostics (exploratory)

See `NOTES_stage4_analysis.md` (not pre-registered). Summary:

- EUPPBench spatial gain of `nn_hybrid_res` over `nn_hybrid` is mostly from a few Alpine valley / far stations (Galtür, Warth, Feldkirch, FR 207460, …) whose attributes sit outside the fold's training range.
- The useful piece is the **ensemble-anchored mean**, not the spread head: it limits unconstrained extrapolation rather than improving in-range stations in general.
- German coverage is denser; the same residual head does not help there. Treat residual anchoring as a safeguard, not a general skill improvement.

## Paper recommendation

1. Treat Stage 4 pilot as a **registered negative result**: climate standardization + neighbour attention does **not** clear the pre-registered bar on either dataset.
2. Lean the paper toward **evaluation / cold-start transfer**: leave-station-out on Germany (535) and EUPPBench (117), comparing embedding strategies, residual anchoring, DRN, GNN, and boosted EMOS.
3. Report baselines honestly: DRN/GNN win on random splits, lose on sparse spatial folds where `emos_bst_gl` and residual/unk nets win.
4. Keep the out-of-range diagnosis as **exploratory** motivation / discussion, not a confirmatory claim.
5. Do **not** promote `samos_attn` as a contribution; optional brief report of the NO-GO and the `samos_mlp` > `samos_attn` ablation.

