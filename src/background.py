"""Does adding normal tissues as background classes stop the model from naming the biopsy organ?

On MET500 the model often calls a breast cancer metastasis in the liver "liver cancer": it reads
the normal liver around the tumour. Here GTEx normal tissues are added to training as extra
classes ("NORMAL_Liver", ...), so organ signal has a class of its own to go to. At prediction
time the normal classes are dropped and the probabilities renormalised over the tumour types.

Design, fixed before looking at the results:
- two models with the same genes and settings: tumours only vs tumours + normal tissues;
- genes = those present in both TCGA and GTEx (a gene missing from GTEx would be filled with a
  constant there, and "constant value" would become a give-away for "normal");
- GTEx samples are quantile-normalised to the TCGA training distribution, like external data;
- MET500 is only used for the final comparison; nothing is tuned on it.

    python -m src.background        # downloads ~670 MB on the first run
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.download import fetch
from src.external import (MET500_TYPES, align, load_cptac, load_met500, quantile_to_reference,
                          reference_profile, score)
from src.models import SEED, evaluate, logistic_pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
EXT = ROOT / "data" / "external"
RESULTS = ROOT / "results"

TOIL = "https://toil-xena-hub.s3.us-east-1.amazonaws.com/download/"
FILES = {
    "gtex_norm_count.tsv.gz": (TOIL, "gtex_RSEM_Hugo_norm_count.gz", 667_188_114),
    "gtex_phenotype.tsv.gz": (TOIL, "GTEX_phenotype.gz", 82_442),
}
PER_TISSUE = 150   # GTEx samples per tissue, so no organ dominates training
NORMAL = "NORMAL_"


class TumourOnly:
    """Wrap a fitted pipeline so that it answers with tumour types only: normal-tissue
    probabilities are dropped and the rest renormalised to sum to one."""

    def __init__(self, model):
        self.model = model
        all_classes = model.classes_
        self.keep = np.array([not c.startswith(NORMAL) for c in all_classes])
        self.classes_ = all_classes[self.keep]

    def predict_proba(self, X):
        p = self.model.predict_proba(X)[:, self.keep]
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def pick_gtex_samples(pheno: pd.DataFrame, per_tissue: int = PER_TISSUE, seed: int = SEED) -> pd.DataFrame:
    """Up to `per_tissue` samples per tissue, one per donor and tissue; cultured cells are not tissue."""
    tissue = pheno[~pheno["body_site_detail (SMTSD)"].str.startswith("Cells")].dropna(subset=["_primary_site"])
    tissue = tissue.drop_duplicates(["_patient", "_primary_site"])
    picked = tissue.sample(frac=1, random_state=seed).groupby("_primary_site").head(per_tissue)
    picked = picked.assign(label=NORMAL + picked["_primary_site"].str.replace(" ", "_"))
    return picked[picked.groupby("label").label.transform("size") >= 30]


def gtex_columns() -> list[str]:
    with gzip.open(EXT / "gtex_norm_count.tsv.gz", "rt") as f:
        return f.readline().rstrip("\n").split("\t")


def read_gtex(samples: list[str], genes: set) -> pd.DataFrame:
    path = EXT / "gtex_norm_count.tsv.gz"
    header = gtex_columns()
    cols = [header[0]] + [s for s in header[1:] if s in set(samples)]
    parts = []
    for chunk in pd.read_csv(path, sep="\t", usecols=cols, index_col=0, chunksize=5000,
                             dtype={c: np.float32 for c in cols[1:]}):
        parts.append(chunk[chunk.index.isin(genes)])
    out = pd.concat(parts)
    return out[~out.index.duplicated()].T


def load_setup() -> SimpleNamespace:
    """TCGA train/val/test on the genes shared with GTEx, plus GTEx normalised to the TCGA scale.
    A fifth of the GTEx donors is held out (`g_test`) and never used for training."""
    for name, (hub, remote, size) in FILES.items():
        fetch(name, remote, size, hub=hub, dest=EXT)

    samples = pd.read_csv(DATA / "samples.csv", dtype=str, keep_default_na=False)
    tcga = pd.read_pickle(DATA / "expression.pkl")

    pheno = pd.read_csv(EXT / "gtex_phenotype.tsv.gz", sep="\t")
    # Some phenotype rows have no expression column; sample only among those that do.
    gtex_meta = pick_gtex_samples(pheno[pheno.Sample.isin(set(gtex_columns()[1:]))])
    gtex = read_gtex(gtex_meta.Sample.tolist(), set(tcga.columns))
    genes = np.array([g for g in tcga.columns if g in set(gtex.columns)])
    print(f"GTEx: {len(gtex)} samples, {gtex_meta.label.nunique()} tissues; genes shared with TCGA: {len(genes)} of {tcga.shape[1]}")

    def part(split):
        s = samples[samples.split == split]
        return tcga.loc[s["sample"], genes].to_numpy(np.float32), s["cancer_type"].to_numpy()

    (Xtr, ytr), (Xva, yva), (Xte, yte) = part("train"), part("val"), part("test")
    train_filled = np.where(np.isnan(Xtr), np.nanmedian(Xtr, axis=0), Xtr)
    # GTEx goes through the same per-sample normalisation as the external cohorts.
    Xg = align(gtex.loc[gtex_meta.Sample], genes)
    shared = ~np.isnan(Xg).all(axis=0)
    Xg[:, shared] = quantile_to_reference(Xg[:, shared], reference_profile(train_filled[:, shared]))

    rng = np.random.default_rng(SEED)
    donors = gtex_meta._patient.unique()
    held = set(rng.choice(donors, size=len(donors) // 5, replace=False))
    return SimpleNamespace(genes=genes, Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, Xte=Xte, yte=yte, train_filled=train_filled,
                           Xg=Xg, yg=gtex_meta.label.to_numpy(), gtex_meta=gtex_meta,
                           g_test=gtex_meta._patient.isin(held).to_numpy())


def load_external(s: SimpleNamespace) -> SimpleNamespace:
    """CPTAC-3 and MET500 on the same genes, quantile-normalised to the TCGA training set."""
    def normalised(expr):
        X = align(expr, s.genes)
        sh = ~np.isnan(X).all(axis=0)
        X[:, sh] = quantile_to_reference(X[:, sh], reference_profile(s.train_filled[:, sh]))
        return X

    met_expr, met_meta = load_met500()
    cp_expr, cp_meta = load_cptac()
    return SimpleNamespace(Xm=normalised(met_expr), met_meta=met_meta, allowed=[MET500_TYPES[c] for c in met_meta.cohort],
                           Xc=normalised(cp_expr), cp_meta=cp_meta)


def evaluate_all(name: str, model, s: SimpleNamespace, e: SimpleNamespace) -> tuple[dict, pd.Series]:
    """TCGA test, CPTAC-3 and MET500 scores, MET500 by biopsy site and liver metastases called LIHC."""
    r = {"tcga_test": evaluate(model, s.Xte, s.yte)}
    pc = model.predict_proba(e.Xc)
    r["cptac3"] = score(model.classes_[pc.argmax(1)], pc, model.classes_, [{t} for t in e.cp_meta.cancer_type])
    pm = model.predict_proba(e.Xm)
    pred = model.classes_[pm.argmax(1)]
    r["met500"] = score(pred, pm, model.classes_, e.allowed)
    hit = np.array([p in a for p, a in zip(pred, e.allowed)])
    d = e.met_meta.assign(correct=hit, predicted=pred)
    r["met500_by_biopsy_site"] = d.groupby("biopsy_tissue").correct.agg(["mean", "count"]).query("count >= 15").round(3).rename(
        columns={"mean": "accuracy", "count": "n"}).reset_index().to_dict("records")
    liver = d.biopsy_tissue == "liver"
    r["met500_liver_called_LIHC"] = int(((d.predicted == "LIHC") & liver & ~d.cohort.eq("HCC")).sum())
    print(f"{name:24s} TCGA test acc {r['tcga_test']['accuracy']:.3f} | CPTAC-3 {r['cptac3']['accuracy']:.3f} | "
          f"MET500 {r['met500']['accuracy']:.3f} (top-3 {r['met500']['top3_accuracy']:.3f}) | liver mets called LIHC {r['met500_liver_called_LIHC']}")
    return r, d.groupby("biopsy_tissue").correct.mean()


def plot_by_site(per_site: dict, report_rows: list, labels: dict, title: str, path: Path):
    """Grouped bars: MET500 accuracy per biopsy site (sites with at least 15 patients) for each model."""
    order = [x["biopsy_tissue"] for x in report_rows]
    n = {x["biopsy_tissue"]: x["n"] for x in report_rows}
    sites = pd.DataFrame(per_site).loc[order]
    colors = ["#9aa5a3", "#0a7466", "#b8600e"]
    width = 0.8 / len(per_site)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(sites))
    for i, name in enumerate(per_site):
        ax.bar(x + (i - (len(per_site) - 1) / 2) * width, sites[name], width, label=labels[name], color=colors[i])
    ax.set_xticks(x, [f"{s}\n(n={n[s]})" for s in sites.index], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("MET500 accuracy")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> int:
    s = load_setup()
    print("fitting: tumours only")
    base = logistic_pipeline(k=5000, C=0.1).fit(s.Xtr, s.ytr)
    print("fitting: tumours + normal tissues")
    bg_full = logistic_pipeline(k=5000, C=0.1).fit(np.vstack([s.Xtr, s.Xg[~s.g_test]]),
                                                   np.concatenate([s.ytr, s.yg[~s.g_test]]))
    bg = TumourOnly(bg_full)
    normal_acc = float((bg_full.predict(s.Xg[s.g_test]) == s.yg[s.g_test]).mean())
    print(f"  held-out GTEx donors: normal tissue named correctly {normal_acc:.3f} (n={s.g_test.sum()})")

    report = {"genes": int(len(s.genes)), "gtex_samples": int(len(s.gtex_meta)), "gtex_tissues": int(s.gtex_meta.label.nunique()),
              "normal_tissue_accuracy_heldout": normal_acc, "models": {}}
    e = load_external(s)
    per_site = {}
    for name, model in (("tumours_only", base), ("with_normal_background", bg)):
        report["models"][name], per_site[name] = evaluate_all(name, model, s, e)

    # How often the full model's single best guess is a normal tissue, and which one.
    top = bg_full.classes_[bg_full.predict_proba(e.Xm).argmax(1)]
    is_normal = np.char.startswith(top.astype(str), NORMAL)
    pairs = pd.Series([f"{b} -> {t}" for b, t in zip(e.met_meta.biopsy_tissue[is_normal], top[is_normal])])
    report["models"]["with_normal_background"]["met500_top_guess_normal"] = {
        "share": float(is_normal.mean()), "by_biopsy_site": pairs.value_counts().loc[lambda x: x > 2].head(10).to_dict()}

    plot_by_site(per_site, report["models"]["tumours_only"]["met500_by_biopsy_site"],
                 {"tumours_only": "tumours only", "with_normal_background": "+ GTEx normal tissues"},
                 "Metastases by biopsy site: does a normal-tissue background help?",
                 RESULTS / "figures" / "background_met500.png")
    (RESULTS / "background.json").write_text(json.dumps(report, indent=2, default=float))
    print("saved background.json and figure")
    return 0


if __name__ == "__main__":
    sys.exit(main())
