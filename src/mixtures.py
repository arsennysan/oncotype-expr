"""Train on tumour + normal-tissue mixtures so that organ signal stops changing the answer.

A metastasis biopsy is a mixture: tumour cells plus the normal tissue of the organ it grew in.
The background-class experiment (src/background.py) could not teach the model to ignore that
normal tissue. Here the model is shown such mixtures directly: each training tumour gets one
copy blended with a random GTEx normal sample, and keeps its tumour label.

Design, fixed before looking at the results:
- mixing happens in linear space, as cells do: log2((1 - a) * tumour + a * normal + 1);
- normal samples come from the GTEx training donors, any of the 26 tissues; a ~ U(0.1, 0.6);
- one mixed copy per training tumour, same genes and settings as the tumours-only model;
- a check that does not use MET500: validation tumours mixed with held-out GTEx donors;
- MET500 is only used for the final comparison; nothing is tuned on it.

    python -m src.mixtures          # needs the GTEx and external files from src.background / src.external
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from src.background import evaluate_all, load_external, load_setup, plot_by_site
from src.models import SEED, logistic_pipeline

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FRACTION = (0.1, 0.6)          # share of normal tissue in a training mixture
CHECK_FRACTIONS = (0.0, 0.2, 0.4, 0.6)


def mix(tumour: np.ndarray, normal: np.ndarray, a) -> np.ndarray:
    """Blend log2(x + 1) profiles in linear space; `a` is the normal share (scalar or per row)."""
    a = np.asarray(a, dtype=np.float32).reshape(-1, 1) if np.ndim(a) else np.float32(a)
    lin_t = np.clip(np.exp2(tumour) - 1, 0, None)
    lin_n = np.clip(np.exp2(normal) - 1, 0, None)
    return np.log2((1 - a) * lin_t + a * lin_n + 1).astype(np.float32)


def mixtures(X: np.ndarray, normals: np.ndarray, rng: np.random.Generator, low: float, high: float) -> np.ndarray:
    """One mixed copy of every row of X with a random normal sample and a random normal share."""
    pick = rng.integers(0, len(normals), size=len(X))
    return mix(X, normals[pick], rng.uniform(low, high, size=len(X)))


def synthetic_check(models: dict, Xva, yva, normals, rng) -> pd.DataFrame:
    """Accuracy on validation tumours blended with held-out normal tissue at fixed shares."""
    pick = rng.integers(0, len(normals), size=len(Xva))
    rows = []
    for a in CHECK_FRACTIONS:
        X = mix(Xva, normals[pick], a)
        for name, model in models.items():
            rows.append({"normal_share": a, "model": name, "accuracy": float((model.predict(X) == yva).mean())})
    return pd.DataFrame(rows)


def paired_test(pred_a, pred_b, allowed) -> dict:
    """Exact McNemar test: of the patients the two models disagree on, does one win more often?"""
    hit_a = np.array([p in s for p, s in zip(pred_a, allowed)])
    hit_b = np.array([p in s for p, s in zip(pred_b, allowed)])
    only_a, only_b = int((hit_a & ~hit_b).sum()), int((~hit_a & hit_b).sum())
    p = binomtest(only_b, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"only_first_correct": only_a, "only_second_correct": only_b, "p_value": float(p)}


def main() -> int:
    s = load_setup()
    rng = np.random.default_rng(SEED)
    normals_train, normals_held = s.Xg[~s.g_test], s.Xg[s.g_test]
    va_filled = np.where(np.isnan(s.Xva), np.nanmedian(s.Xtr, axis=0), s.Xva)

    print("fitting: tumours only")
    base = logistic_pipeline(k=5000, C=0.1).fit(s.Xtr, s.ytr)
    print("fitting: tumours + mixtures")
    Xmix = mixtures(s.train_filled, normals_train, rng, *FRACTION)
    mixed = logistic_pipeline(k=5000, C=0.1).fit(np.vstack([s.Xtr, Xmix]), np.concatenate([s.ytr, s.ytr]))

    models = {"tumours_only": base, "with_mixtures": mixed}
    check = synthetic_check(models, va_filled, s.yva, normals_held, np.random.default_rng(SEED + 1))
    print(check.pivot(index="normal_share", columns="model", values="accuracy").round(3).to_string())

    report = {"design": {"normal_share": FRACTION, "mixed_copies_per_tumour": 1, "normal_pool": "GTEx training donors, 26 tissues"},
              "synthetic_check": check.to_dict("records"), "models": {}}
    e = load_external(s)
    per_site = {}
    for name, model in models.items():
        report["models"][name], per_site[name] = evaluate_all(name, model, s, e)

    report["met500_paired_test"] = paired_test(*(model.predict(e.Xm) for model in models.values()), e.allowed)
    print(f"MET500 paired test: {report['met500_paired_test']}")

    plot_by_site(per_site, report["models"]["tumours_only"]["met500_by_biopsy_site"],
                 {"tumours_only": "tumours only", "with_mixtures": "+ tumour/normal mixtures"},
                 "Metastases by biopsy site: does training on mixtures help?",
                 RESULTS / "figures" / "mixtures_met500.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, color, label in (("tumours_only", "#9aa5a3", "tumours only"), ("with_mixtures", "#0a7466", "+ mixtures")):
        d = check[check.model == name]
        ax.plot(d.normal_share * 100, d.accuracy, marker="o", color=color, linewidth=2, label=label)
    ax.set_xlabel("Share of normal tissue mixed in (%)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Validation tumours + held-out normal tissue")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "mixtures_synthetic.png", dpi=150)

    (RESULTS / "mixtures.json").write_text(json.dumps(report, indent=2, default=float))
    print("saved mixtures.json and figures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
