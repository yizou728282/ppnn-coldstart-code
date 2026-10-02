"""Generate the tables and figures of the cold-start evaluation paper from SAVED results only (no training).

    /workspace/ppnn_venv/bin/python scripts/make_paper_figures.py

Inputs (all produced earlier by the Stage 2-4 pipelines):
  results/stage{2,3}/preds, results/stage4/*/preds (git-ignored): ALL CRPS/calibration values are recomputed from these
      at full precision (no rounding before the final table formatting)
  results/stage4/{german,euppbench}/significance_{random,spatial}.csv  DM/BH/bootstrap tests
  results/stage4/{german,euppbench}/per_fold_*.csv, station_groups_*.csv
  results/stage4/pilot_decision.json
  results/stage{2,3}/summary_*.csv  (seen-station CRPS), folds_*.csv, stations.csv
  results/stage{2,3}/preds, results/stage4/*/preds (git-ignored; needed only for per-station maps and the
      extrapolation diagnostic; if missing, those figures are skipped and a note is written)
Outputs: paper/figures/*.pdf|png, paper/tables/*.tex, paper/figures/derived_numbers.json, paper/figures/MISSING.txt
Every number written here is read or computed from the files above; nothing is typed in by hand.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
FIG = ROOT / "paper/figures"; TAB = ROOT / "paper/tables"
FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
S4 = ROOT / "results/stage4"
OLD = {"german": ROOT / "results/stage2", "euppbench": ROOT / "results/stage3"}
DS_LABEL = {"german": "German", "euppbench": "EUPPBench"}
DS_TEX = {"german": "German", "euppbench": "EUPPBench"}
FT = ["random", "spatial"]

# (id, paper label, group). Labels are the ones used in the paper text; a dagger marks methods declared exploratory
# in the protocols. Code IDs appear only in the repository.
METHODS = [
    ("raw_ensemble_gauss", "Raw ensemble", "ref"),
    ("emos_gl", "EMOS", "emos"),
    ("emos_bst_gl", "Boosted EMOS", "emos"),
    ("nn_unk", "NN-UNK", "emb"),
    ("nn_knn", "NN-kNN", "emb"),
    ("nn_noemb", "NN-NoEmb", "emb"),
    ("nn_attr", "NN-Attr", "emb"),
    ("nn_hybrid", "NN-Hybrid", "emb"),
    ("nn_hybrid_knn", "NN-Hybrid-kNN†", "emb"),
    ("nn_hybrid_res", "NN-Hybrid-Res†", "emb"),
    ("drn_lak", "DRN", "lak"),
    ("gnn_geo", "GNN-Geo", "lak"),
    ("samos_mlp", "SAMOS-MLP", "pilot"),
    ("samos_attn", "SAMOS-Attn", "pilot"),
]
MID = [m for m, _, _ in METHODS]; LAB = {m: l for m, l, _ in METHODS}; GRP = {m: g for m, _, g in METHODS}
TEXLAB = {m: l.replace("†", r"$^\dagger$") for m, l in LAB.items()}
# Okabe-Ito colour-blind-safe palette
OI = {"black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
      "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7", "grey": "#999999"}
GCOL = {"ref": OI["grey"], "emos": OI["blue"], "emb": OI["green"], "lak": OI["vermillion"], "pilot": OI["purple"]}
FOLD_COL = [OI["blue"], OI["orange"], OI["green"], OI["vermillion"], OI["purple"], OI["skyblue"], OI["black"]]
FOLD_MK = ["o", "s", "^", "D", "v", "P", "X"]
NAME_FIX = {"Galtuer": "Galtür", "Clermont-Fd": "Clermont-Fd."}
NETS = [m for m in MID if GRP[m] in ("emb", "lak", "pilot")]
OLD_NETS = ["nn_unk", "nn_knn", "nn_noemb", "nn_attr", "nn_hybrid", "nn_hybrid_knn", "nn_hybrid_res"]
missing, derived = [], {}


def tex_escape(s):
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def TL(m):
    """paper label of a method for LaTeX tables"""
    return TEXLAB.get(m, tex_escape(m))


def fp(p):
    return "$<10^{-15}$" if p == 0 else f"{p:.2g}"


# ---------------------------------------------------------------- exact scores from the stored predictions
def load_all():
    """Recompute all scores from the stored prediction files at full precision (avoids double rounding of the
    summary CSVs). Returns T[(ds, ft)] (DataFrame per method) and D[(ds, ft)] (arrays: crps per row, mu, sigma, meta)."""
    import stage4_evaluate as se  # noqa: E402
    from ppnn_residual.metrics import crps_gaussian_np, summary_metrics  # noqa: E402
    if "nn_hybrid_knn" not in se.OLD_NETS:          # exploratory Stage-2/3 variant stored with the other Stage-2/3 nets
        se.OLD_NETS.append("nn_hybrid_knn")
    T, D = {}, {}
    for ds in OLD:
        meta = np.load(OLD[ds] / "test_meta.npz", allow_pickle=True)
        st = meta["station"].astype(int); y = meta["y"].astype(float)
        day = meta["init"] if "init" in meta else meta["date"]
        for ft in FT:
            folds = pd.read_csv(OLD[ds] / f"folds_{ft}.csv", index_col=0); K = int(folds.fold.max()) + 1
            rf = folds.fold.reindex(st).to_numpy()
            rows, crps, par, single = [], {}, {}, {}
            todo = [(m, [0, 1, 2]) for m in MID]
            if ft == "random":
                todo += [(m + "@10seeds", list(range(10))) for m in OLD_NETS]
            for key, seeds in todo:
                m = key.split("@")[0]
                r = se.load_method(ds, ft, m, seeds, rf, K)
                if r is None:
                    missing.append(f"{ds}/{ft}/{key}: prediction files not found (git-ignored preds/)"); continue
                mu, sg = r["ens"]; met = summary_metrics(mu, sg, y)
                crps[key] = crps_gaussian_np(mu, sg, y); par[key] = (mu, sg)
                row = {"method": key, "n_seeds": len(r["single"]), "crps": met["crps"], "cov80": met["coverage_80"],
                       "spread_skill": float(np.sqrt((sg ** 2).mean()) / met["rmse"]), "bias": met["bias_mean_minus_obs"],
                       "single_seed_mean": np.nan, "single_seed_sd": np.nan}
                if r["single"]:
                    sc = {int(k): float(crps_gaussian_np(*v, y).mean()) for k, v in r["single"].items()}
                    single[key] = sc
                    row["single_seed_mean"], row["single_seed_sd"] = float(np.mean(list(sc.values()))), float(np.std(list(sc.values()), ddof=1))
                rows.append(row)
            T[(ds, ft)] = pd.DataFrame(rows).set_index("method")
            D[(ds, ft)] = {"crps": crps, "par": par, "single": single, "st": st, "y": y, "day": day, "rf": rf, "K": K,
                           "folds": folds}
    derived["exact_scores"] = {f"{ds}:{ft}": T[(ds, ft)].to_dict(orient="index") for (ds, ft) in T}
    derived["single_seed_crps"] = {f"{ds}:{ft}": D[(ds, ft)]["single"] for (ds, ft) in D}
    return T, D


# ---------------------------------------------------------------- tables
def table_main(T):
    cols = [(ds, ft) for ds in OLD for ft in FT]
    best = {c: T[c].loc[[m for m in MID if m in T[c].index and m != "raw_ensemble_gauss"], "crps"].min() for c in cols}
    L = [r"\begin{table*}[!htbp]", r"\centering",
         r"\caption{Mean CRPS (\textdegree C) at held-out stations, pooled over the 7 station-held-out folds (every station is "
         r"held out exactly once). \textbf{All networks: 3-seed ensembles} (seeds 0--2; mean of $\mu$ and $\sigma$); single-seed "
         r"mean $\pm$ sd in parentheses. Bold: lowest CRPS in each original three-seed column (raw ensemble excluded); "
         r"this is descriptive, not a superiority claim. Ten-seed spatial scores and inference are in Table~\ref{tab:s5spatial} "
         r"and Section~\ref{sec:res:s5}. German: test year 2016, lead 48\,h; EUPPBench: "
         r"2017--2018, leads 24--120\,h. $^\dagger$Declared exploratory in the protocols.}",
         r"\label{tab:main}", r"\small", r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{l" + "c" * 4 + "}", r"\toprule",
         r" & \multicolumn{2}{c}{German} & \multicolumn{2}{c}{EUPPBench} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}", r"Method & random & spatial & random & spatial \\", r"\midrule"]
    prev = None
    for m in MID:
        if prev is not None and GRP[m] != prev:
            L.append(r"\addlinespace")
        prev = GRP[m]
        cells = []
        for c in cols:
            if m not in T[c].index:
                cells.append("--"); continue
            r = T[c].loc[m]; v = f"{r.crps:.4f}"
            if abs(r.crps - best[c]) < 1e-12:
                v = r"\textbf{" + v + "}"
            if r.n_seeds > 0 and not np.isnan(r.single_seed_mean):
                v += rf" {{\scriptsize ({r.single_seed_mean:.3f}$\pm${r.single_seed_sd:.3f})}}"
            cells.append(v)
        L.append(TL(m) + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    (TAB / "tab_main_crps.tex").write_text("\n".join(L) + "\n")
    rows = []
    for (ds, ft), t in T.items():
        for m in t.index:
            rows.append({"dataset": ds, "folds": ft, "method": m, **t.loc[m].to_dict()})
    pd.DataFrame(rows).to_csv(TAB / "main_crps_long.csv", index=False, float_format="%.6f")


def table_calib(T):
    cols = [(ds, ft) for ds in OLD for ft in FT]
    L = [r"\begin{table*}[!htbp]", r"\centering",
         r"\caption{Calibration at held-out stations (same 3-seed ensembles as Table~\ref{tab:main}): empirical coverage of "
         r"the central 80\% interval (nominal 0.80) and spread--skill ratio (RMS $\sigma$ / RMSE of $\mu$; 1 = consistent).}",
         r"\label{tab:calib}", r"\small", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{l" + "cc" * 4 + "}", r"\toprule",
         r" & \multicolumn{4}{c}{German} & \multicolumn{4}{c}{EUPPBench} \\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
         r" & \multicolumn{2}{c}{random} & \multicolumn{2}{c}{spatial} & \multicolumn{2}{c}{random} & \multicolumn{2}{c}{spatial} \\",
         r"Method" + " & cov80 & SSR" * 4 + r" \\", r"\midrule"]
    for m in MID:
        cells = []
        for c in cols:
            r = T[c].loc[m] if m in T[c].index else None
            cells += ["--", "--"] if r is None else [f"{r.cov80:.3f}", f"{r.spread_skill:.2f}"]
        L.append(TL(m) + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    (TAB / "tab_calibration.tex").write_text("\n".join(L) + "\n")


def table_seen_gap():
    """Seen- vs new-station CRPS (random folds, 10-seed ensembles for networks; stage-2/3 summaries)."""
    L = [r"\begin{table}[!htbp]", r"\centering",
         r"\caption{Cost of cold start, random folds: CRPS (\textdegree C) on the fold's own training stations (``seen'', "
         r"mean over the 7 folds, test period) versus held-out stations (``new''). Gap = new/seen $-1$. \textbf{Networks: "
         r"10-seed ensembles} (seeds 0--9), the design of the German and EUPPBench protocols; the ``new'' values therefore "
         r"differ slightly from the 3-seed values of Table~\ref{tab:main}. Seen and new scores are not computed on the same "
         r"stations (each station is ``seen'' in 6 of 7 folds).}", r"\label{tab:gap}", r"\small", r"\resizebox{\columnwidth}{!}{%",
         r"\begin{tabular}{lcccccc}", r"\toprule",
         r" & \multicolumn{3}{c}{German} & \multicolumn{3}{c}{EUPPBench} \\", r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         r"Method & seen & new & gap & seen & new & gap \\", r"\midrule"]
    G = {ds: pd.read_csv(OLD[ds] / "summary_random.csv").set_index("method") for ds in OLD}
    gaps = {}
    for m in ["emos_gl", "emos_bst_gl", "nn_unk", "nn_knn", "nn_noemb", "nn_attr", "nn_hybrid", "nn_hybrid_knn", "nn_hybrid_res"]:
        cells = []
        for ds in OLD:
            r = G[ds].loc[m]; g = 100 * (r.crps_primary / r.crps_seen_stations_mean_over_folds - 1)
            gaps[f"{ds}:{m}"] = {"seen": r.crps_seen_stations_mean_over_folds, "new": r.crps_primary, "gap_pct": g}
            cells += [f"{r.crps_seen_stations_mean_over_folds:.3f}", f"{r.crps_primary:.3f}", f"{g:+.1f}\\%"]
        L.append(TL(m) + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    (TAB / "tab_seen_vs_new.tex").write_text("\n".join(L) + "\n")
    derived["seen_vs_new_random_10seeds"] = gaps


def table_tests():
    """Descriptive baseline tests + pilot primary tests (DM, BH, block bootstrap) as stored by stage4_evaluate.py."""
    L = [r"\begin{table*}[!htbp]", r"\centering",
         r"\caption{Diebold--Mariano tests on daily-mean CRPS differences (A $-$ B; negative = A better; 3-seed ensembles), "
         r"Newey--West lag 5, 95\% moving-block bootstrap CI (7-day blocks), Benjamini--Hochberg (BH) at $q=0.05$. The "
         r"DRN/GNN-Geo comparisons are \emph{descriptive} (pre-specified pairs, BH within dataset $\times$ fold type, no "
         r"hypothesis registered); pilot tests use BH over the 6 pre-registered tests per dataset. The tests treat the "
         r"station set as fixed and ignore seed variance (Section~\ref{sec:disc:limits}).}", r"\label{tab:tests}", r"\scriptsize",
         r"\begin{tabular}{lllrrrrc}", r"\toprule",
         r"Data & Folds & A vs B & $\Delta$ & rel.\,\% & DM $p$ & 95\% CI & BH \\", r"\midrule"]
    for ds in OLD:
        for ft in FT:
            s = pd.read_csv(S4 / ds / f"significance_{ft}.csv")
            s = s[s.family.isin(["baselines_descriptive", "pilot_primary"])]
            for _, r in s.iterrows():
                bh = "yes" if str(r.bh_reject_q05) == "True" else "no"
                L.append(f"{DS_TEX[ds]} & {ft} & {TL(r.a)} vs {TL(r.b)} & "
                         f"{r.diff_a_minus_b:+.4f} & {r.rel_diff_pct:+.2f} & {fp(r.dm_p)} & [{r.boot_lo:+.4f}, {r.boot_hi:+.4f}] & {bh} \\\\")
            L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (TAB / "tab_tests.tex").write_text("\n".join(L) + "\n")


def table_primary_hybrid():
    """Pre-registered primary tests of nn_hybrid (Stage 2/3 protocols), incl. the pre-registered station-level DM
    fractions (frac_st_*: share of stations with a BH-significant station-level DM test in favour of / against A)."""
    L = [r"\begin{table*}[!htbp]", r"\centering",
         r"\caption{Pre-registered primary tests of NN-Hybrid (German and EUPPBench protocols). Random folds: 10-seed "
         r"ensembles; spatial folds: 3-seed ensembles. DM test on daily means (Newey--West lag 5), BH over the 6 tests per "
         r"dataset and fold type. ``st.$+$''/``st.$-$'': fraction of stations whose own station-level DM test (BH across "
         r"stations) is significant in favour of / against NN-Hybrid (pre-registered station-level analysis).}",
         r"\label{tab:primary}", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lllrrrrcc}", r"\toprule",
         r"Data & Folds & vs B & $\Delta$ & rel.\,\% & DM $p$ & BH & st.$+$ & st.$-$ \\", r"\midrule"]
    out = {}
    for ds in OLD:
        for ft in FT:
            s = pd.read_csv(OLD[ds] / f"significance_{ft}.csv")
            for _, r in s[s.family == "primary"].iterrows():
                L.append(f"{DS_TEX[ds]} & {ft} & {TL(r.b)} & {r.diff_a_minus_b:+.4f} & {r.rel_diff_pct:+.2f} & {fp(r.dm_p)} & "
                         f"{'yes' if str(r.bh_reject_q05) == 'True' else 'no'} & {r.frac_st_a_better_BH:.2f} & {r.frac_st_a_worse_BH:.2f} \\\\")
            L.append(r"\addlinespace")
            out[f"{ds}:{ft}"] = s[["family", "a", "b", "diff_a_minus_b", "rel_diff_pct", "dm_p", "frac_st_a_better_BH",
                                   "frac_st_a_worse_BH", "bh_reject_q05"]].to_dict(orient="records")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (TAB / "tab_primary_hybrid.tex").write_text("\n".join(L) + "\n")
    derived["stage23_tests_with_station_fractions"] = out


def table_pilot(D):
    P = json.loads((S4 / "pilot_decision.json").read_text())
    L = [r"\begin{table}[!htbp]", r"\centering",
         r"\caption{Pre-registered pilot (SAMOS-Attn) against the best of the three primary comparators on each fold type "
         r"(3-seed ensembles). $\Delta$ = CRPS(SAMOS-Attn) $-$ CRPS(best); negative favours the pilot. GO required a "
         r"BH-significant spatial improvement with bootstrap CI excluding 0 and a same-sign random-fold improvement. Last "
         r"column: single-seed CRPS of SAMOS-Attn for seeds 0/1/2.}", r"\label{tab:pilot}", r"\small", r"\resizebox{\columnwidth}{!}{%",
         r"\begin{tabular}{llccccccc}", r"\toprule",
         r"Data & Folds & best comparator & CRPS best & CRPS pilot & $\Delta$ & DM $p$ & BH & seeds 0/1/2 \\", r"\midrule"]
    for ds in ["german", "euppbench"]:
        for ft in FT:
            d = P[ds][ft]; sc = D[(ds, ft)]["single"]["samos_attn"]
            L.append(f"{DS_TEX[ds]} & {ft} & {TL(d['best_primary_comparator'])} & "
                     f"{d['crps_best']:.4f} & {d['crps_samos_attn']:.4f} & {d['diff']:+.4f} & {d['dm_p']:.2g} & "
                     f"{'yes' if d['bh_reject'] else 'no'} & {'/'.join(f'{sc[k]:.3f}' for k in sorted(sc))} \\\\")
        L.append(rf"\multicolumn{{9}}{{r}}{{Decision: \textbf{{{P[ds]['decision']}}}}} \\")
        L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    (TAB / "tab_pilot.tex").write_text("\n".join(L) + "\n")
    derived["pilot_decision"] = P


def derive_numbers(T, D):
    """Numbers quoted in the text."""
    deg = {}
    for ds in OLD:
        for m in MID:
            a, b = T[(ds, "random")].loc[m, "crps"], T[(ds, "spatial")].loc[m, "crps"]
            deg[f"{ds}:{m}"] = {"random": a, "spatial": b, "rel_change_pct": 100 * (b / a - 1)}
    derived["random_to_spatial_change_3seeds"] = deg
    derived["ranking_3seeds"] = {f"{ds}:{ft}": list(T[(ds, ft)].loc[[m for m in MID if m != "raw_ensemble_gauss"], "crps"].sort_values().index)
                                 for ds in OLD for ft in FT}
    d10 = {f"{ds}:{m}": float(T[(ds, "random")].loc[m + "@10seeds", "crps"] - T[(ds, "random")].loc[m, "crps"])
           for ds in OLD for m in OLD_NETS}
    derived["random_10seed_minus_3seed_crps"] = d10
    derived["random_10seed_minus_3seed_range"] = [min(d10.values()), max(d10.values())]
    rel = {f"{ds}:{ft}:{m}": float(100 * (t.loc[m, "crps"] / t.loc["emos_bst_gl", "crps"] - 1)) for (ds, ft), t in T.items() for m in MID}
    derived["rel_to_emos_bst_gl_pct_3seeds"] = rel
    derived["random_folds_all_networks_rel_to_emos_bst_range_pct"] = [
        min(rel[f"{ds}:random:{m}"] for ds in OLD for m in NETS), max(rel[f"{ds}:random:{m}"] for ds in OLD for m in NETS)]
    wins = {}
    for (ds, ft), d in D.items():
        c, rf, K = d["crps"], d["rf"], d["K"]
        pf = pd.DataFrame({m: [c[m][rf == k].mean() for k in range(K)] for m in c}, index=range(1, K + 1))
        pf.to_csv(TAB / f"per_fold_{ds}_{ft}.csv", float_format="%.6f")
        for a, b in [("drn_lak", "emos_bst_gl"), ("gnn_geo", "emos_bst_gl"), ("drn_lak", "nn_hybrid_res"),
                     ("nn_hybrid_res", "emos_bst_gl"), ("nn_unk", "emos_bst_gl"), ("nn_knn", "emos_bst_gl")]:
            diff = pf[a] - pf[b]
            wins[f"{ds}:{ft}:{a}<{b}"] = {"folds_a_better": int((diff < 0).sum()), "n_folds": K,
                                          "per_fold_diff": {int(k): float(v) for k, v in diff.items()},
                                          "per_fold_diff_min": float(diff.min()), "per_fold_diff_max": float(diff.max())}
        derived.setdefault("per_fold_crps_1based", {})[f"{ds}:{ft}"] = pf.to_dict()
        derived.setdefault("fold_sizes_1based", {})[f"{ds}:{ft}"] = {int(k) + 1: int(v) for k, v in d["folds"].fold.value_counts().sort_index().items()}
    derived["per_fold_wins"] = wins


def extra_descriptive_tests(D):
    """Descriptive (uncorrected) DM tests outside the registered protocol families."""
    from stage4_evaluate import compare  # noqa: E402
    out = {}
    for (ds, ft), pairs in {("euppbench", "spatial"): [("nn_knn", "emos_bst_gl"), ("nn_hybrid_res", "emos_bst_gl"),
                                                      ("nn_unk", "emos_bst_gl"), ("samos_mlp", "emos_bst_gl")],
                            ("german", "spatial"): [("drn_lak", "nn_noemb"), ("drn_lak", "samos_mlp"), ("nn_noemb", "samos_mlp")],
                            ("german", "random"): [("nn_hybrid_res", "samos_mlp"), ("nn_hybrid_res", "nn_noemb")],
                            ("euppbench", "random"): [("nn_hybrid_res", "drn_lak")]}.items():
        d = D[(ds, ft)]
        for a, b in pairs:
            c = compare(d["crps"], d["day"], d["st"], a, b)
            out[f"{ds}:{ft}:{a}-{b}"] = {k: float(c[k]) for k in ["diff_a_minus_b", "rel_diff_pct", "dm_p", "boot_lo", "boot_hi"]}
    derived["EXTRA_descriptive_tests_uncorrected"] = out


def isolated_fifth(D):
    """CRPS in the most isolated fifth of held-out stations (distance to nearest training station of the fold), all
    methods. Quintile edges over the scored stations, as in scripts/stage2_evaluate.py (distance_bins_*.csv)."""
    from stage4_diagnose import station_features  # noqa: E402
    out = {}
    for (ds, ft), d in D.items():
        sf = station_features(d["folds"]); scored = np.unique(d["st"])
        dist = sf.dist_nearest_km.reindex(scored)
        edges = np.unique(np.quantile(dist.values, [0, .2, .4, .6, .8, 1]))
        row_d = sf.dist_nearest_km.reindex(d["st"]).to_numpy()
        msk = row_d >= edges[-2]
        sub = {m: float(c[msk].mean()) for m, c in d["crps"].items() if "@" not in m}
        out[f"{ds}:{ft}"] = {"threshold_km": float(edges[-2]), "n_stations": int(len(np.unique(d["st"][msk]))),
                             "crps": sub, "ranking": sorted(sub, key=sub.get)}
        # stations without a training station within 50 km
        out[f"{ds}:{ft}"]["n_no_train_within_50km"] = int((sf.n_train_50km.reindex(scored) == 0).sum())
        out[f"{ds}:{ft}"]["n_scored"] = int(len(scored))
    derived["isolated_fifth_all_methods"] = out


def sensitivity_11members():
    """EUPPBench test forecasts from members 0-10 instead of 51 (pre-registered Stage-3 sensitivity; Stage-3 methods)."""
    out = {}
    for ft in FT:
        s = pd.read_csv(OLD["euppbench"] / f"sensitivity_11members_{ft}.csv").set_index("method")
        d = s.crps_11_members - s.crps_51_members
        nets = [m for m in s.index if m != "raw_ensemble_gauss"]
        r51 = list(s.loc[nets, "crps_51_members"].sort_values().index); r11 = list(s.loc[nets, "crps_11_members"].sort_values().index)
        out[ft] = {"diff": d.to_dict(), "min_excl_raw": float(d[nets].min()), "max_excl_raw": float(d[nets].max()),
                   "ranking_51": r51, "ranking_11": r11, "ranking_changed": r51 != r11,
                   "swaps": [(a, b) for i, (a, b) in enumerate(zip(r51, r11)) if a != b]}
    derived["sensitivity_11members_euppbench"] = out


def cpu_gpu_difference():
    """Discarded T4 GPU runs (results/stage3/colab_t4_partial) vs the final CPU runs of the same fold/seed/variant."""
    G = OLD["euppbench"] / "colab_t4_partial/preds/random"; C = OLD["euppbench"] / "preds/random"
    if not G.exists():
        missing.append("colab_t4_partial preds missing"); return
    runs = {}
    for f in sorted(G.glob("nn_*.npz")):
        m = f.name.split("_f")[0]
        if m in ("nn_knn", "nn_hybrid_knn"):   # derived from the unk / hybrid runs, not separate trainings
            continue
        a, b = np.load(f), np.load(C / f.name)
        dm = np.abs(a["mu"].astype(float) - b["mu"].astype(float))
        runs[f.name] = {"mean_abs_dmu": float(dm.mean()), "max_abs_dmu": float(dm.max())}
    derived["cpu_vs_t4_gpu_final_predictions"] = {
        "n_runs": len(runs), "mean_over_runs_mean_abs_dmu": float(np.mean([v["mean_abs_dmu"] for v in runs.values()])),
        "max_abs_dmu": float(max(v["max_abs_dmu"] for v in runs.values())),
        "nn_unk_f0_s0": runs.get("nn_unk_f0_s0.npz")}


# ---------------------------------------------------------------- figure: random vs spatial
def fig_random_vs_spatial(T):
    ms = [m for m in MID if m != "raw_ensemble_gauss"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 5.4), sharey=True)
    for ax, ds in zip(axs, OLD):
        y = np.arange(len(ms))[::-1]
        for i, m in zip(y, ms):
            a, b = T[(ds, "random")].loc[m], T[(ds, "spatial")].loc[m]
            ax.plot([a.crps, b.crps], [i, i], color=GCOL[GRP[m]], lw=1.2, alpha=0.7)
            ax.scatter(a.crps, i, marker="o", s=36, facecolor="white", edgecolor=GCOL[GRP[m]], zorder=3)
            ax.scatter(b.crps, i, marker="s", s=36, color=GCOL[GRP[m]], zorder=3)
            for r in (a, b):
                if r.n_seeds > 0 and not np.isnan(r.single_seed_sd):
                    ax.errorbar(r.single_seed_mean, i, xerr=r.single_seed_sd, fmt="none", ecolor="0.6", lw=0.8, zorder=1)
        ax.axvline(T[(ds, "spatial")].loc["emos_bst_gl", "crps"], color=OI["blue"], ls=":", lw=0.9)
        ax.set_yticks(y); ax.set_yticklabels([LAB[m] for m in ms], fontsize=8)
        n = len(np.unique(np.load(OLD[ds] / "test_meta.npz", allow_pickle=True)["station"]))
        ax.set_title(f"{DS_LABEL[ds]} ({n} scored stations)", fontsize=10)
        ax.set_xlabel("CRPS (°C) at held-out stations"); ax.grid(axis="x", alpha=0.3)
    axs[0].scatter([], [], marker="o", facecolor="white", edgecolor="k", label="random folds")
    axs[0].scatter([], [], marker="s", color="k", label="spatial folds")
    axs[0].plot([], [], color="0.6", label="single-seed mean ± sd")
    axs[0].plot([], [], color=OI["blue"], ls=":", label="Boosted EMOS, spatial")
    axs[0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(FIG / f"fig_crps_random_vs_spatial.{ext}", dpi=200)
    plt.close(fig)


def _map_axes(ax, folds):
    ax.set_aspect(1 / np.cos(np.deg2rad(folds.lat.mean())))
    ax.set_xlabel("Longitude (°E)", fontsize=8); ax.set_ylabel("Latitude (°N)", fontsize=8); ax.tick_params(labelsize=7)


def fig_fold_maps():
    fig, axs = plt.subplots(2, 2, figsize=(9, 8.5))
    for j, ds in enumerate(OLD):
        for i, ft in enumerate(FT):
            f = pd.read_csv(OLD[ds] / f"folds_{ft}.csv", index_col=0); ax = axs[i, j]
            for k in range(int(f.fold.max()) + 1):
                g = f[f.fold == k]
                ax.scatter(g.lon, g.lat, c=FOLD_COL[k], marker=FOLD_MK[k], s=11 if ds == "german" else 26,
                           edgecolor="k", linewidth=0.2)
            _map_axes(ax, f); ax.set_title(f"{DS_LABEL[ds]} ({len(f)} stations), {ft} folds", fontsize=9)
    h = [plt.Line2D([], [], marker=FOLD_MK[k], ls="", color=FOLD_COL[k], markeredgecolor="k", markeredgewidth=0.3,
                    label=f"fold {k + 1}") for k in range(7)]
    fig.legend(handles=h, loc="lower center", ncol=7, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ext in ["pdf", "png"]:
        fig.savefig(FIG / f"fig_fold_maps.{ext}", dpi=200)
    plt.close(fig)


def per_station(D, ds, ft, methods):
    d = D[(ds, ft)]
    return pd.DataFrame({m: pd.Series(d["crps"][m]).groupby(d["st"]).mean() for m in methods if m in d["crps"]}), d["folds"]


def fig_diff_maps(D):
    pairs = [("drn_lak", "emos_bst_gl"), ("nn_hybrid", "nn_hybrid_res")]
    names = pd.read_csv(OLD["euppbench"] / "stations.csv", index_col=0)["name"].replace(NAME_FIX)
    fig, axs = plt.subplots(2, 2, figsize=(9, 8.5))
    for j, ds in enumerate(OLD):
        ps, folds = per_station(D, ds, "spatial", ["drn_lak", "emos_bst_gl", "nn_hybrid", "nn_hybrid_res"])
        for i, (a, b) in enumerate(pairs):
            ax = axs[i, j]
            d = (ps[a] - ps[b]); f = folds.loc[d.index]
            lim = float(np.nanquantile(np.abs(d), 0.95))
            sc = ax.scatter(f.lon, f.lat, c=d, cmap="PuOr_r", vmin=-lim, vmax=lim, s=12 if ds == "german" else 28,
                            edgecolor="k", linewidth=0.2)
            for s_ in d.abs().sort_values(ascending=False).index[:3]:
                lab = f"{names.get(s_, s_)} {d[s_]:+.2f}" if ds == "euppbench" else f"{d[s_]:+.2f}"
                ax.annotate(lab, (f.lon[s_], f.lat[s_]), fontsize=6, xytext=(3, 3), textcoords="offset points")
            plt.colorbar(sc, ax=ax, shrink=0.8, extend="both").set_label(f"CRPS {LAB[a]} − {LAB[b]} (°C)", fontsize=7)
            _map_axes(ax, f); ax.set_title(f"{DS_LABEL[ds]}, spatial folds ({len(d)} scored stations)", fontsize=9)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(FIG / f"fig_station_diff_maps_spatial.{ext}", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- figure: extrapolation diagnostic (EXPLORATORY)
def fig_extrapolation(D):
    from stage4_diagnose import station_features  # noqa: E402
    from scipy.stats import spearmanr  # noqa: E402
    names = pd.read_csv(OLD["euppbench"] / "stations.csv", index_col=0)["name"].replace(NAME_FIX)
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    diag = {}
    for j, ds in enumerate(OLD):
        ps, folds = per_station(D, ds, "spatial", ["drn_lak", "emos_bst_gl", "nn_hybrid", "nn_hybrid_res"])
        sf = station_features(folds); aom = (folds.alt - folds.orog)
        for i, (a, b) in enumerate([("nn_hybrid", "nn_hybrid_res"), ("drn_lak", "emos_bst_gl")]):
            ax = axs[i, j]
            d = (ps[a] - ps[b]); s = sf.reindex(d.index); oor = s.any_attr_outside_range.astype(bool)
            ax.scatter(aom[d.index][~oor], d[~oor], s=10, c=OI["grey"], marker="o", label="all attributes inside training range")
            ax.scatter(aom[d.index][oor], d[oor], s=22, c=OI["vermillion"], marker="^", label="some attribute outside range")
            ax.axhline(0, color="k", lw=0.6)
            for st_ in d.abs().sort_values(ascending=False).index[:5]:
                nm = names.get(st_, str(st_)) if ds == "euppbench" else f"{int(folds.alt[st_])} m"
                ax.annotate(nm, (aom[st_], d[st_]), fontsize=6, xytext=(3, 2), textcoords="offset points")
            ax.set_xlabel("Station altitude − model orography (m)", fontsize=8)
            ax.set_ylabel(f"CRPS {LAB[a]} − {LAB[b]} (°C)", fontsize=8)
            ax.set_title(f"{DS_LABEL[ds]}, spatial folds ({len(d)} scored stations)", fontsize=9)
            ax.grid(alpha=0.3)
            rho, p = spearmanr(s.attr_novelty, d)
            diag[f"{ds}:{a}-{b}"] = {"spearman_rho_vs_attr_novelty": float(rho), "p": float(p),
                                    "n_out_of_range": int(oor.sum()), "n": int(len(d)),
                                    "mean_station_diff_out": float(d[oor].mean()), "mean_station_diff_in": float(d[~oor].mean()),
                                    "top5": {str(k): float(d[k]) for k in d.abs().sort_values(ascending=False).index[:5]}}
    axs[0, 0].legend(fontsize=7)
    fig.suptitle("Exploratory (post hoc): per-station CRPS difference vs altitude − orography", fontsize=9)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(FIG / f"fig_extrapolation_diagnostic.{ext}", dpi=200)
    plt.close(fig)
    derived["extrapolation_diagnostic_station_mean_diffs"] = diag


def station_details(D):
    """EXPLORATORY: station-mean CRPS, bias and alt-orog of the most affected EUPPBench stations (spatial folds)."""
    d = D[("euppbench", "spatial")]
    folds = d["folds"]; names = pd.read_csv(OLD["euppbench"] / "stations.csv", index_col=0)["name"].replace(NAME_FIX)
    out = {}
    for nm in ["Galtür", "Warth", "Feldkirch", "Clermont-Fd.", "Oberstdorf"]:
        ids = names.index[names == nm]
        if len(ids) == 0:
            continue
        s = int(ids[0]); msk = d["st"] == s
        rec = {"alt": float(folds.alt[s]), "orog": float(folds.orog[s]), "alt_minus_orog": float(folds.alt[s] - folds.orog[s])}
        for m in ["raw_ensemble_gauss", "emos_bst_gl", "nn_unk", "nn_noemb", "nn_attr", "nn_hybrid", "nn_hybrid_res", "drn_lak", "gnn_geo"]:
            mu = d["par"][m][0][msk]
            rec[m] = {"crps": float(d["crps"][m][msk].mean()), "bias": float((mu - d["y"][msk]).mean())}
        out[nm] = rec
    derived["EXPLORATORY_station_details_euppbench_spatial"] = out


# ---------------------------------------------------------------- EXPLORATORY: in-range vs out-of-range subsets
RANGE_PAIRS = [("drn_lak", "emos_bst_gl"), ("drn_lak", "samos_attn"), ("drn_lak", "nn_hybrid_res"), ("nn_hybrid_res", "emos_bst_gl"),
               ("gnn_geo", "emos_bst_gl"), ("nn_hybrid", "nn_hybrid_res"), ("nn_unk", "emos_bst_gl")]


def exploratory_range_subsets(D):
    """EXPLORATORY (post hoc, not pre-registered). Held-out stations split by whether ANY of (lat, lon, alt, orog,
    alt-orog) lies outside the [min, max] of that fold's training stations (stage4_diagnose.station_features, commit
    3b3e555). DM tests within subsets are uncorrected and descriptive."""
    from stage4_evaluate import compare  # noqa: E402
    from stage4_diagnose import station_features  # noqa: E402
    out = {}
    for (ds, ft), d in D.items():
        crps = {m: c for m, c in d["crps"].items() if "@" not in m and m != "raw_ensemble_gauss"}
        st, day = d["st"], d["day"]
        sf = station_features(d["folds"])
        oor_st = sf.index[sf.any_attr_outside_range.astype(bool)]
        scored = np.unique(st); oor_row = np.isin(st, oor_st)
        trig = {c: int(((sf[f"{c}_excess"] > 0) & sf.index.isin(scored)).sum()) for c in ["lat", "lon", "alt", "orog", "alt_minus_orog"]}
        rec = {"definition": "any of lat, lon, alt, orog, alt-orog outside [min,max] of the fold's training stations",
               "n_stations_scored": int(len(scored)), "n_out": int(np.isin(scored, oor_st).sum()),
               "row_share_out": float(oor_row.mean()), "stations_triggered_by_attribute": trig,
               "out_station_ids": [int(v) for v in scored[np.isin(scored, oor_st)]]}
        for name, msk in [("in", ~oor_row), ("out", oor_row), ("all", np.ones(len(st), bool))]:
            sub = {m: float(c[msk].mean()) for m, c in crps.items()}
            rec[f"ranking_{name}"] = sorted(sub, key=sub.get)
            rec[f"crps_{name}"] = sub
            rec[f"rel_to_emos_bst_gl_pct_{name}"] = {m: 100 * (v / sub["emos_bst_gl"] - 1) for m, v in sub.items()}
            tests = {}
            for a, b in RANGE_PAIRS:
                c = compare({a: crps[a][msk], b: crps[b][msk]}, day[msk], st[msk], a, b)
                ps = pd.DataFrame({"s": st[msk], "a": crps[a][msk], "b": crps[b][msk]}).groupby("s").mean()
                tests[f"{a}-{b}"] = {"diff": float(c["diff_a_minus_b"]), "rel_pct": float(c["rel_diff_pct"]),
                                     "dm_p": float(c["dm_p"]), "boot_lo": float(c["boot_lo"]), "boot_hi": float(c["boot_hi"]),
                                     "frac_stations_a_better": float((ps.a < ps.b).mean()), "n_a_better": int((ps.a < ps.b).sum()),
                                     "n_stations": int(len(ps))}
            rec[f"tests_{name}"] = tests
        share = {}
        for a, b in RANGE_PAIRS:
            dd = crps[a] - crps[b]
            share[f"{a}-{b}"] = {"pooled_diff": float(dd.mean()), "contrib_out": float(dd[oor_row].sum() / len(dd)),
                                 "contrib_in": float(dd[~oor_row].sum() / len(dd))}
        rec["pooled_diff_decomposition"] = share
        out[f"{ds}:{ft}"] = rec
    derived["EXPLORATORY_range_subsets"] = out


def table_groups():
    R = derived["EXPLORATORY_range_subsets"]
    L = [r"\begin{table}[!htbp]", r"\centering",
         r"\caption{\textbf{Exploratory, post hoc.} Spatial folds: CRPS (\textdegree C, 3-seed ensembles) of held-out "
         r"stations whose five attributes (lat, lon, alt, orog, alt$-$orog) all lie inside the [min, max] range of the "
         r"fold's training stations (``in'') or not (``out''). $n$ = number of stations. Bold: lowest in the column; "
         r"in-range differences between the best methods are small and mostly untested (Section~\ref{sec:res:extrap}).}",
         r"\label{tab:groups}", r"\small", r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{lcccc}", r"\toprule",
         r" & \multicolumn{2}{c}{German} & \multicolumn{2}{c}{EUPPBench} \\", r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}"]
    hdr = "Method"
    for ds in OLD:
        r = R[f"{ds}:spatial"]
        hdr += rf" & in ($n$={r['n_stations_scored'] - r['n_out']}) & out ($n$={r['n_out']})"
    L += [hdr + r" \\", r"\midrule"]
    best = {(ds, k): min(R[f"{ds}:spatial"][f"crps_{k}"].values()) for ds in OLD for k in ("in", "out")}
    out = {}
    for m in MID:
        if m == "raw_ensemble_gauss":
            continue
        cells = []
        for ds in OLD:
            r = R[f"{ds}:spatial"]
            for k in ("in", "out"):
                v = r[f"crps_{k}"][m]; t = f"{v:.3f}"
                cells.append(r"\textbf{" + t + "}" if abs(v - best[(ds, k)]) < 1e-12 else t)
            out[f"{ds}:{m}"] = {"in": r["crps_in"][m], "out": r["crps_out"][m]}
        L.append(TL(m) + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    (TAB / "tab_station_groups_spatial.tex").write_text("\n".join(L) + "\n")
    derived["station_groups_spatial"] = out


def table_range_subsets():
    R = derived["EXPLORATORY_range_subsets"]
    L = [r"\begin{table*}[!htbp]", r"\centering",
         r"\caption{\textbf{Exploratory, post hoc.} Pairwise comparisons within the in-range and out-of-range subsets of "
         r"Table~\ref{tab:groups} (spatial folds, 3-seed ensembles). $\Delta$ = CRPS(A) $-$ CRPS(B) (negative: A better); "
         r"DM $p$ on daily means (Newey--West lag 5), \emph{not} corrected for multiplicity and not part of any "
         r"pre-registered family; ``st.'' = fraction of stations in the subset where A has the lower station-mean CRPS "
         r"(descriptive; stations are not independent); ``contrib.'' = contribution of the subset to the pooled "
         r"all-station $\Delta$ (the two contributions sum to the pooled $\Delta$).}",
         r"\label{tab:range}", r"\scriptsize", r"\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{lrrrrrrrrr}", r"\toprule",
         r" & & \multicolumn{4}{c}{in range} & \multicolumn{4}{c}{out of range} \\",
         r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
         r"A vs B & pooled $\Delta$ & $\Delta$ & DM $p$ & st. & contrib. & $\Delta$ & DM $p$ & st. & contrib. \\", r"\midrule"]
    for ds in OLD:
        r = R[f"{ds}:spatial"]
        L.append(rf"\multicolumn{{10}}{{l}}{{\textit{{{DS_TEX[ds]}: {r['n_stations_scored'] - r['n_out']} stations in range, {r['n_out']} out of range}}}} \\")
        for a, b in RANGE_PAIRS:
            k = f"{a}-{b}"; d = r["pooled_diff_decomposition"][k]; ti, to = r["tests_in"][k], r["tests_out"][k]
            L.append(f"{TL(a)} vs {TL(b)} & {d['pooled_diff']:+.3f} & {ti['diff']:+.3f} & {fp(ti['dm_p'])} & "
                     f"{ti['frac_stations_a_better']:.2f} & {d['contrib_in']:+.3f} & {to['diff']:+.3f} & {fp(to['dm_p'])} & "
                     f"{to['frac_stations_a_better']:.2f} & {d['contrib_out']:+.3f} \\\\")
        L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (TAB / "tab_range_subsets_spatial.tex").write_text("\n".join(L) + "\n")


def main():
    T, D = load_all()
    table_main(T); table_calib(T); table_seen_gap(); table_tests(); table_primary_hybrid(); table_pilot(D)
    derive_numbers(T, D); extra_descriptive_tests(D); isolated_fifth(D); sensitivity_11members(); cpu_gpu_difference()
    fig_random_vs_spatial(T); fig_fold_maps(); fig_diff_maps(D); fig_extrapolation(D); station_details(D)
    exploratory_range_subsets(D); table_groups(); table_range_subsets()
    (FIG / "derived_numbers.json").write_text(json.dumps(derived, indent=1, default=float))
    (FIG / "MISSING.txt").write_text("\n".join(missing) + "\n" if missing else "nothing missing\n")
    print("tables:", sorted(p.name for p in TAB.iterdir())); print("figures:", sorted(p.name for p in FIG.iterdir()))
    print("missing:", missing or "none")


if __name__ == "__main__":
    main()
