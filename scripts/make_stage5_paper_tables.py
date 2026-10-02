"""Generate supplementary manuscript tables from the delivered final reports."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results/stage5/eval/final"
TABLES = ROOT / "paper/tables"
LABELS = {
    "raw_ensemble_gauss": "Raw ensemble", "emos_gl": "EMOS", "emos_bst_gl": "Boosted EMOS",
    "emos_nn1": "EMOS-NN1", "emos_idw": "EMOS-IDW", "emos_reg": "EMOS-Reg",
    "emos_regidw": "EMOS-RegIDW", "samos_lin": "SAMOS-Lin", "nn_unk": "NN-UNK",
    "nn_knn": "NN-kNN", "nn_noemb": "NN-NoEmb", "nn_attr": "NN-Attr", "nn_hybrid": "NN-Hybrid",
    "nn_hybrid_knn": r"NN-Hybrid-kNN$^\dagger$", "nn_hybrid_res": r"NN-Hybrid-Res$^\dagger$",
    "drn_lak": "DRN", "gnn_geo": "GNN-Geo", "samos_mlp": "SAMOS-MLP", "samos_attn": "SAMOS-Attn",
}


def read(name):
    with (FINAL / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ci(lo, hi):
    return f"[{float(lo):+.3f}, {float(hi):+.3f}]"


def table(name, label, caption, columns, header, rows):
    text = [r"\begin{table}[htbp]", r"\centering", r"\small", r"\caption{" + caption + "}",
            r"\label{" + label + "}", r"\begin{tabular}{" + columns + "}", r"\toprule",
            header + r" \\", r"\midrule", *[" & ".join(row) + r" \\" for row in rows],
            r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (TABLES / name).write_text("\n".join(text), encoding="utf-8")


def main():
    decisions = json.loads((FINAL / "decisions.json").read_text())
    did_rows = []
    for ds, dslabel in (("euppbench", "EUPPBench"), ("german", "German")):
        for method in ("drn_lak", "gnn_geo"):
            d = decisions[ds][method]
            did_rows.append([dslabel, LABELS[method], f"{d['random_F_minus_E']:+.3f}",
                            f"{d['spatial_F_minus_E']:+.3f}", f"{d['did']:+.3f}", ci(d["lo"], d["hi"])])
    table("tab_stage5_did.tex", "tab:s5did",
          r"Supplementary difference-in-differences (DiD) relative to Boosted EMOS (E), in degrees C. "
          r"Positive DiD means a loss of relative skill under spatial blocking. Spatial ensembles use 10 seeds; "
          r"the DRN and GNN-Geo random ensembles retain 3 seeds. Intervals resample stations and seeds jointly.",
          "llrrrr", r"Dataset & F & Random F$-$E & Spatial F$-$E & DiD & 95\% CI", did_rows)
    summaries = {ds: {r["method"]: r for r in read(f"summary_{ds}_spatial.csv")}
                 for ds in ("german", "euppbench")}
    table("tab_stage5_spatial.tex", "tab:s5spatial",
          r"Final supplementary spatial-fold CRPS (degrees C), at the five-decimal precision of the delivered reports. "
          r"All networks use 10 seeds (0--9); statistical "
          r"methods are deterministic. Scores are descriptive; numerical ordering alone is not evidence of a difference. "
          r"$^\dagger$Exploratory variants. The original three-seed comparison is retained in Table~\ref{tab:main}.",
          "lrr", "Method & German & EUPPBench",
          [[label, f"{float(summaries['german'][m]['crps']):.5f}",
            f"{float(summaries['euppbench'][m]['crps']):.5f}"] for m, label in LABELS.items()])
    partlabels = {"part_km7_s2027": "$K=7$, seed 2027", "part_km7_s2028": "$K=7$, seed 2028",
                  "part_km7_s2029": "$K=7$, seed 2029", "part_km5_s2026": "$K=5$, seed 2026",
                  "part_km10_s2026": "$K=10$, seed 2026", "part_loco": "Leave one country out"}
    parts = []
    for part, label in partlabels.items():
        a, b = (decisions["euppbench"][m]["partitions"][part] for m in ("drn_lak", "gnn_geo"))
        parts.append([label, f"{a['did']:+.3f}", ci(a["lo"], a["hi"]),
                      f"{b['did']:+.3f}", ci(b["lo"], b["hi"])])
    table("tab_stage5_partitions.tex", "tab:s5parts",
          r"DiD relative to Boosted EMOS under the six additional EUPPBench spatial partitions. "
          r"All network ensembles in these comparisons use 3 seeds. Units: degrees C; intervals are seed-nested "
          r"station-bootstrap 95\% intervals. These partitions share observations and are not independent replications.",
          "lrrrr", r"Partition & DRN & 95\% CI & GNN-Geo & 95\% CI", parts)
    m1a = []
    for ft in ("spatial", "random"):
        # Pair reports use six significant digits; the summaries preserve five
        # decimal places. Use the latter to avoid rounding the pair value twice.
        summary = {r["method"]: r for r in read(f"summary_euppbench_{ft}.csv")}
        for row in read(f"pairs_euppbench_{ft}.csv"):
            if not row["a"].endswith("_fcavail"):
                continue
            m1a.append([ft.capitalize(), LABELS[row["a"].removesuffix("_fcavail")],
                        f"{float(row['crps_b']):.5f}", f"{float(summary[row['a']]['crps']):.5f}",
                        f"{float(row['diff']):+.4f}", ci(row["st_nested_lo"], row["st_nested_hi"])])
    table("tab_stage5_m1a.tex", "tab:s5m1a",
          r"EUPPBench sensitivity to defining neighbour availability using forecasts only. Three-seed ensembles "
          r"are compared with their original three-seed versions. $\Delta$ is corrected minus original CRPS "
          r"(degrees C); intervals include station and seed resampling. Original results remain primary.",
          "llrrrr", r"Folds & Method & Original & Corrected & $\Delta$ & 95\% CI", m1a)
    bst = {ft: {r["m"]: r for r in read(f"bstsens_euppbench_{ft}.csv")} for ft in ("spatial", "random")}
    table("tab_stage5_boosting.tex", "tab:s5boost",
          r"EUPPBench Boosted-EMOS stopping sensitivity. CRPS is in degrees C; the spatial spread--skill ratio "
          r"(SSR) is also shown. The registered 1000-iteration setting remains primary; test scores are not used "
          r"to choose the stopping point. AIC and validation select separately within each fold and lead.",
          "lrrr", "Stopping & Random CRPS & Spatial CRPS & Spatial SSR",
          [[{"aic": "AIC", "val": "Validation"}.get(m, m), f"{float(bst['random'][m]['crps']):.5f}",
            f"{float(bst['spatial'][m]['crps']):.5f}", f"{float(bst['spatial'][m]['spread_skill']):.3f}"]
           for m in ("100", "250", "500", "1000", "2000", "4000", "aic", "val")])
    changes = {}
    for ft in ("random", "spatial"):
        logs = [json.loads((ROOT / f"results/stage5/euppbench/m1a/logs/{ft}/gnn_geo_fcavail_f{k}_s0.json").read_text())
                for k in range(7)]
        changes[ft] = sum(x["n_heldout_test_rows"] * x["frac_heldout_test_cases_neighbour_set_changed"] for x in logs) / sum(x["n_heldout_test_rows"] for x in logs)
    numbers = {"source_commit": "8a7c56d", "decisions": decisions, "spatial_summary": summaries,
               "gnn_fraction_test_neighbour_sets_changed": changes,
               "neighbour_audit": json.loads((ROOT / "handoff/STAGE5_NEIGHBOUR_AUDIT.json").read_text())}
    (ROOT / "paper/figures/stage5_numbers.json").write_text(json.dumps(numbers, indent=2) + "\n")
    print(json.dumps({"tables_generated": 5, "gnn_neighbour_change_fractions": changes}, indent=2))


if __name__ == "__main__":
    main()
