"""Write the CSV summary and LaTeX table for the reviewer-requested DiD intervals.
Reads only results/review_supplementary/did_<dataset>.json written by analysis/review_supplementary_did.py.

  python analysis/make_review_supplementary_table.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
IN = REPO / "results/review_supplementary"
LABEL = {"primary_10v3_drn": ("DRN", "10/3 (primary)"), "primary_10v3_gnn": ("GNN-Geo", "10/3 (primary)"),
         "matched_3v3_drn": ("DRN", "3/3 (matched)"), "matched_3v3_gnn": ("GNN-Geo", "3/3 (matched)"),
         "forecast_only_3v3_gnn": ("GNN-Geo FO", "3/3 (matched)")}
DSN = {"euppbench": "EUPPBench", "german": "German"}


def fmt(x, nd=4):
    """Signed fixed-point number for LaTeX, with a typographic minus sign."""
    return f"{x:+.{nd}f}".replace("-", "$-$")


def main():
    rows = []
    for ds in ("euppbench", "german"):
        r = json.loads((IN / f"did_{ds}.json").read_text())
        assert r["all_checks_passed"]
        for c, v in r["contrasts"].items():
            for scheme, b in v.items():
                rows.append({"dataset": ds, "contrast": c, "resampling": scheme,
                             "n_clusters": r["n_stations"] if scheme == "station" else r["block_schemes"][scheme]["n_clusters"],
                             "did": b["did"], "lo": b["lo"], "hi": b["hi"], "width": b["hi"] - b["lo"],
                             "excludes_zero": bool(b["lo"] > 0 or b["hi"] < 0), "p_did_gt0": b["p_did_gt0"],
                             "n_seeds_spatial": v["station"]["n_seeds_spatial"], "n_seeds_random": v["station"]["n_seeds_random"]})
    df = pd.DataFrame(rows)
    df.to_csv(IN / "did_review_supplementary.csv", index=False, float_format="%.6f")

    # Table A: matched-seed and forecast-only DiD with the primary (station+seed) resampling
    a = df[df.resampling == "station"]
    lines = [r"\begin{table*}[htbp]", r"\singlespacing", r"\centering", r"\small",
             r"\caption{Post hoc DiD intervals relative to Boosted EMOS (\textdegree C). "
             r"Intervals use the primary seed-nested station bootstrap (2000 replicates; the same shared station weights and seed-resampling streams "
             r"as the primary analysis). Seeds 10/3 is the primary configuration; 3/3 rows restrict spatial ensembles to seeds 0--2 (matched). Primary rows reproduce Table~3 (main article). "
             r"FO: forecast-only neighbour availability. P is the bootstrap fraction of replicates with DiD $>0$.}",
             r"\label{tab:review:matched}", r"\begin{tabular}{lllrrr}", r"\toprule",
             r"Data & Method & Seeds sp/rd & DiD & 95\% CI & P \\", r"\midrule"]
    for _, x in a.iterrows():
        m, s = LABEL[x.contrast]
        lines.append(f"{DSN[x.dataset]} & {m} & {s.split(' (')[0]} & {fmt(x.did)} & [{fmt(x.lo)}, {fmt(x.hi)}] & {x.p_did_gt0:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (REPO / "paper/tables/tab_review_matched_did.tex").write_text("\n".join(lines) + "\n")

    # Table B: paired regional block bootstrap
    order = ["station", "primary_folds_7", "countries_5"] + [s for s in df.resampling.unique() if s.startswith("kmeans")]
    name = lambda s, n: "Stations (primary)" if s == "station" else ("Primary spatial folds" if s.startswith("primary") else
                                                                      ("Countries" if s.startswith("countries") else "$k$-means regions"))
    lines = []
    for ds, label in (("euppbench", "tab:review:block"), ("german", "tab:review:blockgerman")):
        cap = (f"Paired regional block bootstrap of the {DSN[ds]} DiD relative to Boosted EMOS (\\textdegree C). "
               r"In each of 2000 replicates whole clusters of stations are resampled with replacement and the same cluster weights are applied "
               r"to the spatial and random designs, nested with the primary seed resampling. $G$ is the number of clusters; "
               r"$k$-means regions use the repository's station-coordinate $k$-means (seed 2026). "
               + (r"FO: forecast-only neighbour availability. " if ds == "euppbench" else "")
               + r"Bootstrap intervals with few, unequal clusters are approximate.")
        lines += [r"\begin{table*}[htbp]", r"\singlespacing", r"\centering", r"\small", "\\caption{" + cap + "}",
                  "\\label{" + label + "}", r"\begin{tabular}{llrrr}", r"\toprule",
                  r"Contrast & Resampling unit & $G$ & 95\% CI & P \\", r"\midrule"]
        first = True
        for c in [k for k in LABEL if k in set(df[df.dataset == ds].contrast)]:
            sub = df[(df.dataset == ds) & (df.contrast == c)].set_index("resampling")
            if not first:
                lines.append(r"\addlinespace")
            first = False
            m, s = LABEL[c]
            for sch in [o for o in order if o in sub.index]:
                x = sub.loc[sch]
                if sch == "station":
                    lines.append(f"\\multicolumn{{5}}{{l}}{{{m}, seeds {s}; DiD {fmt(x.did)}}} \\\\")
                lines.append(f" & {name(sch, x.n_clusters)} & {int(x.n_clusters)} & [{fmt(x.lo)}, {fmt(x.hi)}] & {x.p_did_gt0:.3f} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (REPO / "paper/tables/tab_review_block_did.tex").write_text("\n".join(lines) + "\n")
    write_section(df)
    print(df.to_string())


def write_section(df):
    """Draft supplementary subsection; every number is taken from the computed CSV."""
    g = lambda ds, c, r="station": df[(df.dataset == ds) & (df.contrast == c) & (df.resampling == r)].iloc[0]
    ci = lambda x: f"[{fmt(x.lo)}, {fmt(x.hi)}]\\degC"
    unit = lambda r: {"primary_folds_7": "primary folds", "countries_5": "countries"}.get(r, r.replace("kmeans_", "$k$-means, $G=$"))
    name_of = lambda x: f"{LABEL[x.contrast][0]} ({LABEL[x.contrast][1].split(' (')[1].rstrip(')')}), {unit(x.resampling)}: {ci(x)}"
    block = df[df.resampling != "station"]
    wid = df[df.resampling == "station"].set_index(["dataset", "contrast"]).width
    ratio = {ds: b.width.values / np.array([wid[(ds, c)] for c in b.contrast]) for ds, b in block.groupby("dataset")}
    eb, gb = block[block.dataset == "euppbench"], block[block.dataset == "german"]
    eb_in = "; ".join(name_of(x) for _, x in eb[~eb.excludes_zero].iterrows())
    gm, gp = gb[gb.contrast.str.startswith("matched")], gb[gb.contrast.str.startswith("primary")]
    gm_in = "; ".join(name_of(x) for _, x in gm[~gm.excludes_zero].iterrows())
    gp_ex = "; ".join(name_of(x) for _, x in gp[gp.excludes_zero].iterrows())
    d, gg, fo = g("euppbench", "matched_3v3_drn"), g("euppbench", "matched_3v3_gnn"), g("euppbench", "forecast_only_3v3_gnn")
    gd, ggn = g("german", "matched_3v3_drn"), g("german", "matched_3v3_gnn")
    pd_, pg = g("german", "primary_10v3_drn"), g("german", "primary_10v3_gnn")
    eu_all = all(x.excludes_zero for x in (d, gg, fo))
    ge_m = all(x.excludes_zero for x in (gd, ggn)); ge_p = not any(x.excludes_zero for x in (pd_, pg))
    t = [r"\subsection{Matched-seed, forecast-only and regional-block DiD intervals}\label{sec:supp:review}",
         r"These post hoc analyses use only preserved per-seed predictions; no model was retrained. They reuse the primary "
         r"seed-nested station bootstrap (2000 replicates) with the same station weights and seed-resampling streams; the "
         r"primary DiD intervals of Table~3 (main article) are reproduced exactly. Matched contrasts restrict the spatial ensembles to "
         r"seeds 0--2, so both designs use three-seed ensembles (Table~\ref{tab:review:matched})."]
    t.append(f"On EUPPBench the matched DRN DiD is {fmt(d.did)}\\degC\\ {ci(d)} and the matched GNN-Geo DiD is "
             f"{fmt(gg.did)}\\degC\\ {ci(gg)}. The forecast-only GNN-Geo DiD is {fmt(fo.did)}\\degC\\ {ci(fo)}. "
             + (r"All three station-and-seed intervals exclude zero, so in these stored predictions the direction of relative "
                r"deterioration under spatial blocking does not depend on the ensemble-size mismatch or on observation-based "
                r"neighbour selection. The intervals overlap substantially with the primary ones and do not separate the "
                r"magnitudes of the configurations." if eu_all else r"Not all of these intervals exclude zero."))
    t.append(f"On the German data the matched DiDs are DRN {fmt(gd.did)}\\degC\\ {ci(gd)} "
             f"(primary {fmt(pd_.did)}\\degC\\ {ci(pd_)}) and GNN-Geo {fmt(ggn.did)}\\degC\\ {ci(ggn)} "
             f"(primary {fmt(pg.did)}\\degC\\ {ci(pg)}). "
             + (r"Both matched German intervals exclude zero, whereas both primary ones include it. The German finding that the "
                r"DiD intervals include zero therefore depends on comparing ten spatial with three random seeds; with matched "
                r"three-seed ensembles a small positive German DiD is also detected. This further weakens any contrast between the "
                r"two datasets that rests on one significant and one nonsignificant interval." if (ge_m and ge_p) else ""))
    t.append(r"Tables~\ref{tab:review:block} and~\ref{tab:review:blockgerman} replace station resampling with a paired regional block bootstrap: whole clusters of "
             r"stations are resampled and the same cluster weights are applied to both designs, nested with the seed resampling. "
             r"Clusters are the seven primary spatial folds, the five countries (EUPPBench) and $k$-means regions on station "
             r"coordinates. "
             f"For EUPPBench, block intervals are {ratio['euppbench'].min():.1f}--{ratio['euppbench'].max():.1f} times as wide as the "
             f"corresponding station intervals; {int(eb.excludes_zero.sum())} of {len(eb)} exclude zero"
             + (f", the exceptions being GNN-Geo {eb_in}. " if eb_in else ". ").replace("GNN-Geo GNN-Geo", "GNN-Geo")
             + r"Several lower bounds lie close to zero, and with five to 30 unequal clusters these percentile intervals are "
             r"approximate. The regional analysis therefore supports the sign of the EUPPBench DiD less firmly than station "
             r"resampling and leaves its magnitude poorly determined. "
             f"German block intervals are {ratio['german'].min():.1f}--{ratio['german'].max():.1f} times as wide as the station "
             f"intervals; {len(gp) - int(gp.excludes_zero.sum())} of {len(gp)} primary German block intervals include zero"
             + (f" (exceptions: {gp_ex})" if gp_ex else "")
             + f", and {int(gm.excludes_zero.sum())} of {len(gm)} matched German block intervals exclude zero"
             + (f", the exception being {gm_in}." if gm_in else "."))
    t.append(r"\input{tables/tab_review_matched_did}")
    t.append(r"\input{tables/tab_review_block_did}")
    (REPO / "paper/review_supplementary_section.tex").write_text("\n\n".join(t) + "\n")


if __name__ == "__main__":
    main()
