"""Show what the model says about one tumour: the true type, its top three guesses and the
genes behind the answer.

    python -m src.predict                         # one random tumour from the test set
    python -m src.predict --random 3              # three random test tumours
    python -m src.predict --split metastasis      # a random metastasis
    python -m src.predict TCGA-A7-A0CE-01         # a specific tumour by its TCGA id

Needs the files made by `python -m src.prepare` and `python -m src.train`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.interpret import KNOWN_MARKERS, class_contributions

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"

CANCER_NAMES = {
    "ACC": "Adrenocortical carcinoma", "BLCA": "Bladder urothelial carcinoma", "BRCA": "Breast invasive carcinoma",
    "CESC": "Cervical squamous cell carcinoma and endocervical adenocarcinoma", "CHOL": "Cholangiocarcinoma",
    "COAD": "Colon adenocarcinoma", "DLBC": "Diffuse large B-cell lymphoma", "ESCA": "Esophageal carcinoma",
    "GBM": "Glioblastoma", "HNSC": "Head and neck squamous cell carcinoma", "KICH": "Kidney chromophobe",
    "KIRC": "Kidney renal clear cell carcinoma", "KIRP": "Kidney renal papillary cell carcinoma",
    "LGG": "Brain lower grade glioma", "LIHC": "Liver hepatocellular carcinoma", "LUAD": "Lung adenocarcinoma",
    "LUSC": "Lung squamous cell carcinoma", "MESO": "Mesothelioma", "OV": "Ovarian serous cystadenocarcinoma",
    "PAAD": "Pancreatic adenocarcinoma", "PCPG": "Pheochromocytoma and paraganglioma",
    "PRAD": "Prostate adenocarcinoma", "READ": "Rectum adenocarcinoma", "SARC": "Sarcoma",
    "SKCM": "Skin cutaneous melanoma", "STAD": "Stomach adenocarcinoma", "TGCT": "Testicular germ cell tumours",
    "THCA": "Thyroid carcinoma", "THYM": "Thymoma", "UCEC": "Uterine corpus endometrial carcinoma",
    "UCS": "Uterine carcinosarcoma", "UVM": "Uveal melanoma",
}
SPLIT_NOTE = {
    "test": "test set: the model has never seen it",
    "val": "validation set: used to choose the model, not to train it",
    "metastasis": "metastasis: the model learned on primary tumours only",
    "train": "training set: the model learned from this tumour, so this is not a fair test",
}


def top_guesses(proba: np.ndarray, classes: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
    """The k most likely classes for one sample, most likely first."""
    order = np.argsort(proba)[::-1][:k]
    return [(str(classes[i]), float(proba[i])) for i in order]


def explain(pipe, x: np.ndarray, genes: np.ndarray, cancer_type: str, n: int = 8) -> pd.DataFrame:
    """Genes that pushed one sample's score towards `cancer_type` the most.
    push = coefficient x standardised expression (exact SHAP value for this model);
    z = expression in standard deviations from the training average."""
    contrib, names = class_contributions(pipe, x.reshape(1, -1), genes)
    ci = list(pipe.named_steps["clf"].classes_).index(cancer_type)
    coef = pipe.named_steps["clf"].coef_[ci]
    push = contrib[0, :, ci]
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(coef != 0, push / coef, 0.0)
    order = np.argsort(push)[::-1][:n]
    markers = set(KNOWN_MARKERS.get(cancer_type, []))
    return pd.DataFrame({"gene": names[order], "push": push[order], "z": z[order],
                         "known_marker": [g in markers for g in names[order]]})


def describe(sample: str, split: str, truth: str, pipe, x: np.ndarray, genes: np.ndarray, n_genes: int) -> str:
    proba = pipe.predict_proba(x.reshape(1, -1))[0]
    guesses = top_guesses(proba, pipe.classes_)
    lines = [f"Tumour {sample} ({SPLIT_NOTE.get(split, split)})",
             f"True type: {truth:5s} {CANCER_NAMES.get(truth, '')}", "", "Model's top 3 guesses:"]
    for rank, (c, p) in enumerate(guesses, start=1):
        mark = "  <- correct" if c == truth else ""
        lines.append(f"  {rank}. {c:5s} {CANCER_NAMES.get(c, ''):45s} {p:6.1%}{mark}")
    predicted = guesses[0][0]
    lines.append("")
    lines.append("RIGHT: the top guess is the true type." if predicted == truth else
                 "WRONG: the top guess is not the true type." +
                 (f" The true type is guess #{[c for c, _ in guesses].index(truth) + 1}." if truth in dict(guesses) else
                  " The true type is not in the top 3."))
    # Explain the top guess, and the true type too when the model got it wrong.
    for t in [predicted] + ([truth] if predicted != truth and truth in pipe.classes_ else []):
        table = explain(pipe, x, genes, t, n_genes)
        lines += ["", f"Genes that pushed the model towards {t} (push > 0 = towards {t}):",
                  "  gene          push   expression vs average"]
        for row in table.itertuples():
            note = "  (known marker)" if row.known_marker else ""
            lines.append(f"  {row.gene:12s} {row.push:+6.2f}   {row.z:+5.1f} SD{note}")
    return "\n".join(lines)


def load_model():
    bundle = joblib.load(DATA / "best_model.joblib")
    if "scale" not in bundle["model"].named_steps:
        # Gene-level explanations need the logistic model; train.py always saves it.
        bundle = joblib.load(DATA / "logistic_model.joblib")
    return bundle["model"], np.asarray(bundle["genes"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Show the model's answer for one or more TCGA tumours.")
    ap.add_argument("samples", nargs="*", help="TCGA sample ids such as TCGA-A7-A0CE-01")
    ap.add_argument("--random", type=int, default=1, help="how many random tumours to show (default 1)")
    ap.add_argument("--split", default="test", choices=["test", "val", "metastasis", "train"],
                    help="where random tumours come from (default: test)")
    ap.add_argument("--genes", type=int, default=8, help="how many genes to list (default 8)")
    ap.add_argument("--seed", type=int, default=None, help="fix the random choice to repeat it")
    args = ap.parse_args(argv)

    if not (DATA / "best_model.joblib").exists():
        print("No trained model yet. Run first:\n  python -m src.download\n  python -m src.prepare\n  python -m src.train")
        return 1
    samples = pd.read_csv(DATA / "samples.csv", dtype=str, keep_default_na=False).set_index("sample")
    if args.samples:
        unknown = [s for s in args.samples if s not in samples.index]
        if unknown:
            examples = ", ".join(samples[samples.split == "test"].index[:3])
            print(f"Unknown sample id(s): {', '.join(unknown)}\nExamples from the test set: {examples}")
            return 1
        chosen = args.samples
    else:
        pool = samples.index[samples.split == args.split]
        chosen = list(np.random.default_rng(args.seed).choice(pool, size=min(args.random, len(pool)), replace=False))

    pipe, genes = load_model()
    expr = pd.read_pickle(DATA / "expression.pkl")
    for i, s in enumerate(chosen):
        if i:
            print("\n" + "-" * 72 + "\n")
        x = expr.loc[s, genes].to_numpy(np.float32)
        print(describe(s, samples.at[s, "split"], samples.at[s, "cancer_type"], pipe, x, genes, args.genes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
