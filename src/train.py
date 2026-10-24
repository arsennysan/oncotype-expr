"""Train and compare classifiers.

Models are tuned and chosen on the validation set. The test set is scored once, at the end,
for the chosen model and for the reference models, so its numbers are never used for decisions.

    python -m src.train
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, confusion_matrix

from src.models import evaluate, lightgbm_pipeline, logistic_pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"


def load():
    samples = pd.read_csv(DATA / "samples.csv")
    expr = pd.read_pickle(DATA / "expression.pkl")
    primary = samples[samples["split"] != "metastasis"]
    parts = {}
    for name in ("train", "val", "test"):
        s = primary[primary["split"] == name]
        parts[name] = (expr.loc[s["sample"]].to_numpy(np.float32), s["cancer_type"].to_numpy())
    return parts, expr.columns.to_numpy()


def fit(name, model, X, y):
    t = time.time()
    model.fit(X, y)
    print(f"  fitted {name} in {time.time() - t:.0f} s")
    return model


def main() -> int:
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    parts, genes = load()
    Xtr, ytr = parts["train"]
    Xva, yva = parts["val"]
    Xte, yte = parts["test"]
    print(f"train {len(ytr)}, val {len(yva)}, test {len(yte)} tumours, {len(genes)} genes, {len(set(ytr))} types")

    candidates = {"majority class": DummyClassifier(strategy="most_frequent")}
    for C in (0.01, 0.1, 1.0):
        candidates[f"logistic C={C}"] = logistic_pipeline(k=5000, C=C)
    candidates["lightgbm"] = lightgbm_pipeline(k=2000)

    val_scores = {}
    fitted = {}
    for name, model in candidates.items():
        fitted[name] = fit(name, model, Xtr, ytr)
        val_scores[name] = evaluate(model, Xva, yva)
        print(f"  {name:18s} val macro-F1 {val_scores[name]['macro_f1']:.3f}")

    best = max((n for n in val_scores if n != "majority class"), key=lambda n: val_scores[n]["macro_f1"])
    best_logistic = max((n for n in val_scores if n.startswith("logistic")), key=lambda n: val_scores[n]["macro_f1"])
    print(f"chosen on validation: {best}")

    report = {"chosen_model": best, "validation": val_scores, "test": {}}
    for name in dict.fromkeys(["majority class", best_logistic, "lightgbm", best]):
        report["test"][name] = evaluate(fitted[name], Xte, yte)
        print(f"  TEST {name:18s} " + "  ".join(f"{k} {v:.3f}" for k, v in report["test"][name].items()))
    (RESULTS / "metrics.json").write_text(json.dumps(report, indent=2))

    model = fitted[best]
    pred = model.predict(Xte)
    per_class = pd.DataFrame(classification_report(yte, pred, output_dict=True, zero_division=0)).T
    per_class.round(3).to_csv(RESULTS / "per_class_test.csv")

    labels = model.classes_
    cm = confusion_matrix(yte, pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(cm, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel("Predicted cancer type")
    ax.set_ylabel("True cancer type")
    ax.set_title(f"Test set confusion matrix, {best} (row-normalised)")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "confusion_matrix.png", dpi=150)

    joblib.dump({"model": model, "genes": genes, "name": best}, DATA / "best_model.joblib")
    joblib.dump({"model": fitted[best_logistic], "genes": genes, "name": best_logistic}, DATA / "logistic_model.joblib")
    print(f"saved metrics, per-class report, confusion matrix and models")
    return 0


if __name__ == "__main__":
    sys.exit(main())
