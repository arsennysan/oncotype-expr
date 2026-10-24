"""External validation: score the TCGA-trained model on cohorts it has never seen.

- CPTAC-3: prospective primary tumours collected by the Clinical Proteomic Tumor Analysis
  Consortium, other hospitals and a different RNA-seq pipeline (GDC STAR, log2(TPM + 1)).
- MET500: metastatic biopsies from the University of Michigan (Robinson et al., Nature 2017),
  log2(TPM + 0.001), with the primary site of each patient's cancer.

Every external dataset is on its own scale, so each sample is quantile-normalised to the TCGA
training distribution on its own (no information from the rest of the cohort is used), and
genes missing from the external data are filled with the TCGA training median by the model's
own imputer. Raw values are scored too, to show why this step matters.

    python -m src.external          # downloads ~950 MB on the first run
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.download import fetch

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
EXT = ROOT / "data" / "external"
RESULTS = ROOT / "results"

GDC = "https://gdc-hub.s3.us-east-1.amazonaws.com/download/"
UCSC = "https://ucscpublic.xenahubs.net/download/"
FILES = {
    "cptac3_tpm.tsv.gz": (GDC, "CPTAC-3.star_tpm.tsv.gz", 896_275_745),
    "cptac3_clinical.tsv.gz": (GDC, "CPTAC-3.clinical.tsv.gz", 369_405),
    "gencode_v36.probemap": (GDC, "gencode.v36.annotation.gtf.gene.probemap", 3_206_945),
    "met500_log2tpm.txt.gz": (UCSC, "MET500/geneExpression/M.mx.log2.txt.gz", 45_780_273),
    "met500_meta.txt": (UCSC, "MET500/geneExpression/M.meta.plus.txt", 102_957),
}

# CPTAC-3 diagnoses -> TCGA cancer type, by (primary site, primary diagnosis). Anything else
# (oncocytoma, gliosarcoma, rare histologies) has no clean TCGA counterpart and is left out.
CPTAC_TYPES = {
    ("Uterus, NOS", "Endometrioid adenocarcinoma, NOS"): "UCEC",
    ("Kidney", "Renal cell carcinoma, NOS"): "KIRC",      # the CPTAC clear-cell renal cohort
    ("Kidney", "Papillary renal cell carcinoma"): "KIRP",
    ("Brain", "Glioblastoma"): "GBM",
    ("Bronchus and lung", "Adenocarcinoma, NOS"): "LUAD",
    ("Bronchus and lung", "Squamous cell carcinoma, NOS"): "LUSC",
    ("Pancreas", "Infiltrating duct carcinoma, NOS"): "PAAD",
    ("Other and ill-defined sites", "Squamous cell carcinoma, NOS"): "HNSC",  # head and neck cohort
}

# MET500 cohorts -> the TCGA types a correct answer may name. Organ-level labels (lung, colon,
# kidney) accept any TCGA type of that organ; MISC and SECR are mixed groups and are left out.
MET500_TYPES = {
    "ACC": {"ACC"}, "BLCA": {"BLCA"}, "BRCA": {"BRCA"}, "CHOL": {"CHOL"}, "ESCA": {"ESCA"},
    "GBM": {"GBM"}, "HNSC": {"HNSC"}, "OV": {"OV"}, "PAAD": {"PAAD"}, "PRAD": {"PRAD"},
    "SARC": {"SARC"}, "SKCM": {"SKCM"}, "STAD": {"STAD"}, "TGCT": {"TGCT"}, "THCA": {"THCA"},
    "THYM": {"THYM"}, "HCC": {"LIHC"}, "COLO": {"COAD", "READ"}, "LUNG": {"LUAD", "LUSC"},
    "KDNY": {"KIRC", "KIRP", "KICH"},
}


def quantile_to_reference(X: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Replace each sample's values by the reference distribution, keeping the sample's own
    gene ranking. `reference` is the sorted mean TCGA quantile profile over the same genes;
    ties get the mean of the reference values they span. NaN stays NaN."""
    out = np.full(X.shape, np.nan, dtype=np.float32)
    grid = np.linspace(0, 1, len(reference))
    for i, row in enumerate(X):
        ok = ~np.isnan(row)
        ranks = pd.Series(row[ok]).rank(method="average").to_numpy()
        pos = (ranks - 1) / max(ok.sum() - 1, 1)
        out[i, ok] = np.interp(pos, grid, reference)
    return out


def reference_profile(X_train: np.ndarray) -> np.ndarray:
    """Mean of the per-sample sorted values: the 'typical' TCGA sample's distribution."""
    return np.sort(X_train, axis=1).mean(axis=0)


def ensembl_to_symbol(ids: pd.Index, probemap: pd.DataFrame) -> pd.Series:
    """Map versioned Ensembl ids to gene symbols using the version-free id."""
    table = probemap.assign(base=probemap["id"].str.split(".").str[0]).drop_duplicates("base").set_index("base")["gene"]
    return pd.Series(ids.str.split(".").str[0].map(table).to_numpy(), index=ids)


def collapse_to_symbols(matrix: pd.DataFrame, symbols: pd.Series) -> pd.DataFrame:
    """genes x samples (Ensembl) -> samples x genes (symbols); duplicates keep the highest mean."""
    m = matrix.assign(_sym=symbols.reindex(matrix.index).to_numpy()).dropna(subset=["_sym"])
    m = m.assign(_mean=m.drop(columns="_sym").mean(axis=1)).sort_values("_mean", ascending=False)
    return m.drop_duplicates("_sym").set_index("_sym").drop(columns="_mean").T


def score(pred: np.ndarray, proba: np.ndarray, classes: np.ndarray, allowed: list[set]) -> dict:
    """Accuracy and top-3 accuracy where each sample may accept several TCGA types."""
    top3 = classes[np.argsort(proba, axis=1)[:, ::-1][:, :3]]
    hit = np.array([p in a for p, a in zip(pred, allowed)])
    hit3 = np.array([bool(set(t) & a) for t, a in zip(top3, allowed)])
    return {"n": int(len(pred)), "accuracy": float(hit.mean()), "top3_accuracy": float(hit3.mean())}


def load_cptac() -> tuple[pd.DataFrame, pd.DataFrame]:
    clin = pd.read_csv(EXT / "cptac3_clinical.tsv.gz", sep="\t", low_memory=False)
    clin = clin[clin["sample_type.samples"] == "Primary Tumor"].copy()
    clin["cancer_type"] = [CPTAC_TYPES.get(k) for k in zip(clin["primary_site"], clin["primary_diagnosis.diagnoses"])]
    print(f"  CPTAC-3 primary tumours: {len(clin)}, with a TCGA counterpart: {clin.cancer_type.notna().sum()}")
    clin = clin.dropna(subset=["cancer_type"])
    # One tumour per patient, like the TCGA side.
    clin = clin.sort_values("sample").drop_duplicates("case_id")
    with gzip.open(EXT / "cptac3_tpm.tsv.gz", "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
    keep = [header[0]] + [s for s in header[1:] if s in set(clin["sample"])]
    expr = pd.read_csv(EXT / "cptac3_tpm.tsv.gz", sep="\t", usecols=keep, index_col=0, dtype={s: np.float32 for s in keep[1:]})
    probemap = pd.read_csv(EXT / "gencode_v36.probemap", sep="\t")
    expr = collapse_to_symbols(expr, ensembl_to_symbol(expr.index, probemap))
    meta = clin.set_index("sample").loc[expr.index, ["cancer_type"]]
    return expr, meta


def load_met500() -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = pd.read_csv(EXT / "met500_meta.txt", sep="\t")
    meta = meta[meta.cohort.isin(MET500_TYPES)].copy()
    # Several patients have both a poly-A and an exome-capture library; poly-A is closer to TCGA.
    meta["polyA"] = meta.Sample_id.str.contains("-poly-")
    meta = meta.sort_values(["sample_source", "polyA"], ascending=[True, False]).drop_duplicates("sample_source")
    expr = pd.read_csv(EXT / "met500_log2tpm.txt.gz", sep="\t", index_col=0,
                       usecols=["sample", *meta.Sample_id], dtype={s: np.float32 for s in meta.Sample_id})
    probemap = pd.read_csv(EXT / "gencode_v36.probemap", sep="\t")
    expr = collapse_to_symbols(expr, ensembl_to_symbol(expr.index, probemap))
    meta = meta.set_index("Sample_id").loc[expr.index, ["cohort", "biopsy_tissue", "polyA", "tc"]]
    return expr, meta


def align(expr: pd.DataFrame, genes: np.ndarray) -> np.ndarray:
    """Columns in the model's gene order; genes the cohort lacks become NaN (imputed by the model)."""
    return expr.reindex(columns=genes).to_numpy(np.float32)


def evaluate_cohort(name, model, genes, expr, allowed, X_train, used) -> tuple[dict, np.ndarray]:
    X_raw = align(expr, genes)
    covered = ~np.isnan(X_raw[:, used]).all(axis=0)
    print(f"  {name}: {len(expr)} samples, {covered.sum()} of {len(used)} model genes present")
    # Normalise over the genes both datasets share, then hand the full matrix to the model.
    shared = ~np.isnan(X_raw).all(axis=0)
    X_norm = X_raw.copy()
    X_norm[:, shared] = quantile_to_reference(X_raw[:, shared], reference_profile(X_train[:, shared]))
    out = {"n_samples": int(len(expr)), "model_genes_present": int(covered.sum()), "model_genes": int(len(used))}
    for label, X in (("raw", X_raw), ("quantile_normalised", X_norm)):
        proba = model.predict_proba(X)
        pred = model.classes_[proba.argmax(axis=1)]
        out[label] = score(pred, proba, model.classes_, allowed)
        print(f"    {label:20s} accuracy {out[label]['accuracy']:.3f}  top-3 {out[label]['top3_accuracy']:.3f}")
    return out, pred


def plot_cptac_confusion(truth, pred, path: Path):
    rows = sorted(set(truth))
    cols = rows + sorted(set(pred) - set(rows))
    cm = pd.DataFrame(confusion_matrix(truth, pred, labels=cols)[: len(rows)], index=rows, columns=cols)
    cm = cm.loc[:, cm.sum() > 0]
    share = cm.div(cm.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(1 + 0.55 * cm.shape[1], 1 + 0.55 * cm.shape[0]))
    ax.imshow(share, cmap="Greens", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm.iat[i, j]:
                ax.text(j, i, cm.iat[i, j], ha="center", va="center", fontsize=8,
                        color="white" if share.iat[i, j] > 0.6 else "black")
    ax.set_xticks(range(cm.shape[1]), cm.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(cm.shape[0]), cm.index, fontsize=8)
    ax.set_xlabel("Predicted (TCGA type)")
    ax.set_ylabel("CPTAC-3 diagnosis")
    ax.set_title("External cohort CPTAC-3 (quantile-normalised)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> int:
    EXT.mkdir(parents=True, exist_ok=True)
    for name, (hub, remote, size) in FILES.items():
        fetch(name, remote, size, hub=hub, dest=EXT)

    bundle = joblib.load(DATA / "best_model.joblib")
    model, genes = bundle["model"], np.asarray(bundle["genes"])
    used = model.named_steps["genes"].selected_
    samples = pd.read_csv(DATA / "samples.csv", dtype=str, keep_default_na=False)
    tcga = pd.read_pickle(DATA / "expression.pkl")
    X_train = tcga.loc[samples.loc[samples.split == "train", "sample"], genes].to_numpy(np.float32)
    X_train = np.where(np.isnan(X_train), np.nanmedian(X_train, axis=0), X_train)
    report = {"model": bundle["name"]}

    print("CPTAC-3")
    expr, meta = load_cptac()
    report["cptac3"], pred = evaluate_cohort("CPTAC-3", model, genes, expr, [{t} for t in meta.cancer_type], X_train, used)
    per_type = pd.DataFrame({"cancer_type": meta.cancer_type.to_numpy(), "correct": pred == meta.cancer_type.to_numpy()})
    report["cptac3"]["per_type"] = per_type.groupby("cancer_type").correct.agg(["sum", "count"]).rename(
        columns={"sum": "correct", "count": "n"}).reset_index().to_dict("records")
    errs = pd.Series([f"{t} -> {p}" for t, p in zip(meta.cancer_type, pred) if t != p]).value_counts()
    report["cptac3"]["top_errors"] = errs.head(10).to_dict()
    plot_cptac_confusion(meta.cancer_type.to_numpy(), pred, RESULTS / "figures" / "external_cptac3.png")

    print("MET500")
    expr, meta = load_met500()
    allowed = [MET500_TYPES[c] for c in meta.cohort]
    report["met500"], pred = evaluate_cohort("MET500", model, genes, expr, allowed, X_train, used)
    hit = np.array([p in a for p, a in zip(pred, allowed)])
    d = meta.assign(correct=hit, predicted=pred)
    report["met500"]["per_cohort"] = d.groupby("cohort").correct.agg(["sum", "count"]).rename(
        columns={"sum": "correct", "count": "n"}).reset_index().to_dict("records")
    report["met500"]["by_biopsy_site"] = d.groupby("biopsy_tissue").correct.agg(["mean", "count"]).query("count >= 15").round(3).rename(
        columns={"mean": "accuracy", "count": "n"}).reset_index().to_dict("records")
    report["met500"]["by_library"] = d.groupby("polyA").correct.agg(["mean", "count"]).round(3).rename(
        columns={"mean": "accuracy", "count": "n"}).reset_index().to_dict("records")
    # A metastasis in the liver called liver cancer: the model read the organ, not the tumour.
    report["met500"]["predicted_as_biopsy_organ"] = (
        d[~d.correct].assign(organ=lambda x: x.biopsy_tissue.map({"liver": "LIHC", "lung": "LUAD", "brain": "LGG"}))
        .pipe(lambda x: {"errors": int(len(x)), "named_the_biopsy_organ": int(((x.organ == x.predicted) |
              ((x.biopsy_tissue == "lung") & x.predicted.isin(["LUAD", "LUSC"])) |
              ((x.biopsy_tissue == "brain") & x.predicted.isin(["LGG", "GBM"]))).sum())}))
    report["met500"]["top_errors"] = pd.Series([f"{c} -> {p}" for c, p, h in zip(d.cohort, pred, hit) if not h]).value_counts().head(12).to_dict()

    (RESULTS / "external.json").write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps({k: v for k, v in report["met500"].items() if k not in ("per_cohort",)}, indent=1, default=float))
    print("saved external.json and figure")
    return 0


if __name__ == "__main__":
    sys.exit(main())
