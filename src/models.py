"""Model pipelines and metrics. Every preprocessing step lives inside a scikit-learn Pipeline,
so it is fitted on the training data only and simply applied to validation and test data."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, top_k_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


class TopVarianceGenes(BaseEstimator, TransformerMixin):
    """Keep the k genes with the highest variance in the data it is fitted on."""

    def __init__(self, k: int = 5000):
        self.k = k

    def fit(self, X, y=None):
        var = np.asarray(X).var(axis=0)
        self.selected_ = np.sort(np.argsort(var)[::-1][: self.k])
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.selected_]


def logistic_pipeline(k: int = 5000, C: float = 0.1) -> Pipeline:
    return Pipeline([
        ("genes", TopVarianceGenes(k)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=3000, class_weight="balanced")),
    ])


def lightgbm_pipeline(k: int = 2000, **params) -> Pipeline:
    from lightgbm import LGBMClassifier

    defaults = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                    colsample_bytree=0.3, class_weight="balanced", random_state=SEED, n_jobs=-1, verbose=-1)
    return Pipeline([("genes", TopVarianceGenes(k)), ("clf", LGBMClassifier(**(defaults | params)))])


def evaluate(model, X, y) -> dict:
    """Metrics that stay honest under class imbalance: macro-F1 and balanced accuracy weigh
    every cancer type equally; top-3 accuracy asks whether the right answer is among three guesses."""
    proba = model.predict_proba(X)
    classes = model.classes_
    pred = classes[proba.argmax(axis=1)]
    return {
        "macro_f1": f1_score(y, pred, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "accuracy": accuracy_score(y, pred),
        "top3_accuracy": top_k_accuracy_score(y, proba, k=3, labels=classes),
    }
