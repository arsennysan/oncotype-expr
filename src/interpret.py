"""Which genes drive each prediction, and are they the ones pathologists already know?

For the logistic regression model, SHAP values are exact and cheap: each gene's contribution to
a class score is its coefficient times the gene's standardised expression (relative to the
training mean). We average, for every cancer type, the contributions on that type's test tumours.

    python -m src.interpret
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

# Established tissue / lineage markers (immunohistochemistry and textbook biology), fixed before
# looking at the model. Genes absent from the model's gene set are reported, not silently skipped.
# SILV is the older symbol of PMEL (melanocyte protein gp100), the name used in this dataset.
KNOWN_MARKERS = {
    "LIHC": ["ALB", "APOA2", "FGB", "APOH"],
    "THCA": ["TG", "TPO"],
    "PRAD": ["KLK3", "KLK2", "NKX3-1"],
    "BRCA": ["GATA3", "ESR1", "TFF1"],
    "SKCM": ["MLANA", "SILV", "DCT"],
    "UVM": ["MLANA", "SILV", "DCT"],
    "COAD": ["CDX2", "CDX1", "SATB2"],
    "LUAD": ["SFTPB", "NAPSA"],
    "LUSC": ["KRT5", "TP63", "SOX2"],
    "HNSC": ["KRT5", "TP63"],
    "KIRC": ["CA9", "PAX8"],
    "KIRP": ["PAX8"],
    "OV": ["PAX8", "WT1"],
    "UCEC": ["PGR", "ESR1", "PAX8"],
    "LGG": ["GFAP"],
    "GBM": ["GFAP"],
    "TGCT": ["POU5F1", "NANOG"],
    "PCPG": ["TH", "CHGA", "DBH", "PNMT"],
    "ACC": ["STAR", "NR5A1", "CYP17A1"],
    "DLBC": ["MS4A1", "CD19", "CD79A"],
    "THYM": ["CD1A", "CD3E"],
    "MESO": ["MSLN", "UPK3B"],
    "BLCA": ["UPK1B", "UPK3A", "UPK2", "GATA3"],
    "PAAD": ["PRSS1", "CPA2"],
}


def class_contributions(pipe, X: np.ndarray, genes: np.ndarray):
    """Per-sample, per-gene, per-class contributions of a fitted logistic pipeline.
    Returns (contributions[n, g, c], selected gene names)."""
    Xi = pipe.named_steps["impute"].transform(X)
    sel = pipe.named_steps["genes"]
    Z = pipe.named_steps["scale"].transform(sel.transform(Xi))
    coef = pipe.named_steps["clf"].coef_          # (classes, genes)
    # Z is centred on the training mean, so coef * Z equals SHAP values with the training
    # data as background (the linear-model case of shap.LinearExplainer).
    return Z[:, :, None] * coef.T[None, :, :], genes[sel.selected_]


def top_genes_per_type(contrib, gene_names, y, classes, n=15) -> pd.DataFrame:
    rows = []
    for ci, c in enumerate(classes):
        mean = contrib[y == c, :, ci].mean(axis=0)
        order = np.argsort(mean)[::-1]
        for rank, gi in enumerate(order[:n], start=1):
            rows.append({"cancer_type": c, "rank": rank, "gene": gene_names[gi], "mean_shap": round(float(mean[gi]), 4)})
    return pd.DataFrame(rows)


def marker_check(contrib, gene_names, y, classes, markers=KNOWN_MARKERS) -> pd.DataFrame:
    """Rank of each known marker among all model genes for its own cancer type (1 = most important)."""
    idx = {g: i for i, g in enumerate(gene_names)}
    rows = []
    for c, genes in markers.items():
        if c not in classes:
            continue
        ci = list(classes).index(c)
        mean = contrib[y == c, :, ci].mean(axis=0)
        ranks = (-mean).argsort().argsort() + 1
        for g in genes:
            rows.append({"cancer_type": c, "marker": g, "in_model": g in idx,
                         "rank": int(ranks[idx[g]]) if g in idx else None, "of": len(gene_names)})
    return pd.DataFrame(rows)


def plot_top_genes(top: pd.DataFrame, markers: dict, types: list[str], path: Path, n=8):
    fig, axes = plt.subplots(4, 4, figsize=(15, 13))
    for ax, t in zip(axes.flat, types):
        d = top[(top.cancer_type == t) & (top["rank"] <= n)].iloc[::-1]
        known = set(markers.get(t, []))
        colors = ["#b8600e" if g in known else "#0a7466" for g in d.gene]
        ax.barh(d.gene, d.mean_shap, color=colors)
        ax.set_title(t, fontsize=11, weight="bold")
        ax.tick_params(labelsize=8)
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Top genes behind each prediction (mean SHAP on that type's test tumours)\n"
                 "orange = established marker of the tissue", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> int:
    bundle = joblib.load(DATA / "best_model.joblib")
    pipe, genes = bundle["model"], bundle["genes"]
    if "scale" not in pipe.named_steps:
        print(f"chosen model {bundle['name']} is not the logistic pipeline; using the logistic model")
        bundle = joblib.load(DATA / "logistic_model.joblib")
        pipe, genes = bundle["model"], bundle["genes"]
    samples = pd.read_csv(DATA / "samples.csv", dtype=str, keep_default_na=False)
    expr = pd.read_pickle(DATA / "expression.pkl")
    test = samples[samples.split == "test"]
    X, y = expr.loc[test["sample"]].to_numpy(np.float32), test["cancer_type"].to_numpy()
    classes = pipe.named_steps["clf"].classes_

    contrib, names = class_contributions(pipe, X, genes)
    top = top_genes_per_type(contrib, names, y, classes)
    top.to_csv(RESULTS / "top_genes_per_type.csv", index=False)
    check = marker_check(contrib, names, y, classes)
    check.to_csv(RESULTS / "marker_check.csv", index=False)

    found = check.dropna(subset=["rank"])
    print(f"known markers in the model's gene set: {len(found)} of {len(check)}")
    print(f"  in the top 10 for their tissue: {(found['rank'] <= 10).sum()}")
    print(f"  in the top 50: {(found['rank'] <= 50).sum()}")
    print("best rank per cancer type:")
    print(found.sort_values("rank").groupby("cancer_type").first()[["marker", "rank"]].sort_values("rank").to_string())
    print("not in the model gene set:", ", ".join(check.loc[~check.in_model, "marker"].unique()))

    show = ["LIHC", "THCA", "PRAD", "BRCA", "SKCM", "COAD", "LUAD", "LUSC",
            "KIRC", "OV", "LGG", "TGCT", "PCPG", "ACC", "DLBC", "BLCA"]
    plot_top_genes(top, KNOWN_MARKERS, show, RESULTS / "figures" / "shap_top_genes.png")
    print("saved top genes, marker check and figure")
    return 0


if __name__ == "__main__":
    sys.exit(main())
