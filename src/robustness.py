"""Does the model hold up outside the conditions it was trained in?

1. Metastases: the model learned on primary tumours; can it name the origin of a metastasis?
2. Unseen collection sites: train on some hospitals, test on hospitals the model never saw,
   and compare with an ordinary random split of the same size.
3. Gene budget: how few genes keep most of the performance?

The test set stays untouched here: checks 2 and 3 use the training and validation tumours only.

    python -m src.robustness
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models import SEED, logistic_pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
C = 0.1  # regularisation chosen on the validation set in src/train.py


def metastases(model, expr, samples) -> dict:
    met = samples[samples.split == "metastasis"]
    X, y = expr.loc[met["sample"]].to_numpy(np.float32), met["cancer_type"].to_numpy()
    proba = model.predict_proba(X)
    classes = model.classes_
    pred = classes[proba.argmax(axis=1)]
    met = met.assign(predicted=pred)
    melanoma = met.cancer_type == "SKCM"
    out = {
        "n": len(met),
        "accuracy": accuracy_score(y, pred),
        "top3_accuracy": top_k_accuracy_score(y, proba, k=3, labels=classes),
        "melanoma": {"n": int(melanoma.sum()), "accuracy": float((met.predicted == met.cancer_type)[melanoma].mean())},
        "other": {"n": int((~melanoma).sum()), "accuracy": float((met.predicted == met.cancer_type)[~melanoma].mean())},
        "other_cases": met.loc[~melanoma, ["sample", "cancer_type", "predicted"]].to_dict("records"),
        "melanoma_errors": met.loc[melanoma & (met.predicted != met.cancer_type), "predicted"].value_counts().to_dict(),
    }
    return out


def cross_validate(X, y, folds, label) -> dict:
    """Fit on each training fold, score on its held-out fold. Held-out tumours of a cancer type
    absent from the training fold cannot be predicted at all and are counted separately."""
    preds, truth, skipped = [], [], 0
    for k, (tr, te) in enumerate(folds, start=1):
        model = logistic_pipeline(k=5000, C=C).fit(X[tr], y[tr])
        seen = np.isin(y[te], model.classes_)
        skipped += int((~seen).sum())
        preds.append(model.predict(X[te][seen]))
        truth.append(y[te][seen])
        print(f"  {label}: fold {k} done")
    p, t = np.concatenate(preds), np.concatenate(truth)
    return {"macro_f1": f1_score(t, p, average="macro"), "accuracy": accuracy_score(t, p),
            "scored": int(len(t)), "skipped_unseen_type": skipped}


def unseen_sites(expr, samples) -> dict:
    dev = samples[samples.split.isin(["train", "val"])].reset_index(drop=True)
    X, y, sites = expr.loc[dev["sample"]].to_numpy(np.float32), dev["cancer_type"].to_numpy(), dev["site"].to_numpy()
    by_site = list(GroupKFold(n_splits=5).split(X, y, groups=sites))
    random = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y))
    return {"n_tumours": len(dev), "n_sites": int(pd.Series(sites).nunique()),
            "random_split": cross_validate(X, y, random, "random split"),
            "unseen_sites": cross_validate(X, y, by_site, "unseen sites")}


def gene_budget(expr, samples) -> pd.DataFrame:
    """Rank genes by their largest absolute coefficient in a model fitted on the training set,
    then refit with only the top k and score on the validation set."""
    tr = samples[samples.split == "train"]
    va = samples[samples.split == "val"]
    Xtr, ytr = expr.loc[tr["sample"]].to_numpy(np.float32), tr["cancer_type"].to_numpy()
    Xva, yva = expr.loc[va["sample"]].to_numpy(np.float32), va["cancer_type"].to_numpy()
    full = logistic_pipeline(k=5000, C=C).fit(Xtr, ytr)
    selected = full.named_steps["genes"].selected_
    importance = np.abs(full.named_steps["clf"].coef_).max(axis=0)
    ranked = selected[np.argsort(importance)[::-1]]
    rows = []
    for k in (10, 25, 50, 100, 250, 500, 1000, 2500, 5000):
        cols = ranked[:k]
        m = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                      ("clf", LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))]).fit(Xtr[:, cols], ytr)
        rows.append({"genes": k, "val_macro_f1": f1_score(yva, m.predict(Xva[:, cols]), average="macro")})
        print(f"  gene budget {k:5d}: val macro-F1 {rows[-1]['val_macro_f1']:.3f}")
    return pd.DataFrame(rows)


def plot_budget(df: pd.DataFrame, path: Path):
    best = df.val_macro_f1.max()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df.genes, df.val_macro_f1, marker="o", color="#0a7466", linewidth=2)
    ax.axhline(0.9 * best, color="#b8600e", linestyle="--", linewidth=1)
    ax.text(df.genes.iloc[0], 0.9 * best + 0.01, "90% of the best", color="#b8600e", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Number of genes (ranked by model weight, log scale)")
    ax.set_ylabel("Validation macro-F1")
    ax.set_title("How many genes are enough?")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> int:
    # Keep codes as text: one TCGA collection site is literally "NA", which pandas would read as missing.
    samples = pd.read_csv(DATA / "samples.csv", dtype=str, keep_default_na=False)
    expr = pd.read_pickle(DATA / "expression.pkl")
    bundle = joblib.load(DATA / "best_model.joblib")
    report = {"model": bundle["name"]}

    print("1. metastases")
    report["metastases"] = metastases(bundle["model"], expr, samples)
    m = report["metastases"]
    print(f"  accuracy {m['accuracy']:.3f}, top-3 {m['top3_accuracy']:.3f}; melanoma {m['melanoma']['accuracy']:.3f} "
          f"(n={m['melanoma']['n']}), other {m['other']['accuracy']:.3f} (n={m['other']['n']})")
    print(f"  melanoma errors: {m['melanoma_errors']}")

    print("2. unseen collection sites")
    report["sites"] = unseen_sites(expr, samples)
    for k in ("random_split", "unseen_sites"):
        r = report["sites"][k]
        print(f"  {k:13s} macro-F1 {r['macro_f1']:.3f}  accuracy {r['accuracy']:.3f}  (skipped {r['skipped_unseen_type']})")

    print("3. gene budget")
    budget = gene_budget(expr, samples)
    budget.to_csv(RESULTS / "gene_budget.csv", index=False)
    plot_budget(budget, RESULTS / "figures" / "gene_budget.png")
    report["gene_budget"] = budget.to_dict("records")

    (RESULTS / "robustness.json").write_text(json.dumps(report, indent=2, default=float))
    print("saved robustness.json, gene_budget.csv and figure")
    return 0


if __name__ == "__main__":
    sys.exit(main())
