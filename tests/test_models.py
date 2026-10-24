"""Tests for the model building blocks on synthetic data."""
import numpy as np

from src.models import TopVarianceGenes, evaluate, logistic_pipeline


def test_top_variance_genes_is_fitted_on_training_data_only():
    rng = np.random.default_rng(0)
    train = rng.normal(size=(50, 6)) * np.array([1, 5, 1, 4, 1, 1])  # genes 1 and 3 vary most
    test = rng.normal(size=(20, 6)) * np.array([9, 1, 9, 1, 9, 9])   # a different pattern
    sel = TopVarianceGenes(k=2).fit(train)
    assert sel.selected_.tolist() == [1, 3]
    assert sel.transform(test).shape == (20, 2)
    assert np.array_equal(sel.transform(test), test[:, [1, 3]])


def test_logistic_pipeline_learns_separable_classes_and_metrics_are_sane():
    rng = np.random.default_rng(1)
    centers = rng.normal(scale=4, size=(3, 30))
    y = np.repeat(["A", "B", "C"], [60, 40, 20])
    X = centers[np.searchsorted(["A", "B", "C"], y)] + rng.normal(size=(120, 30))
    model = logistic_pipeline(k=20, C=1.0).fit(X, y)
    m = evaluate(model, X, y)
    assert m["macro_f1"] > 0.95
    assert m["top3_accuracy"] == 1.0
    assert set(m) == {"macro_f1", "balanced_accuracy", "accuracy", "top3_accuracy"}


def test_pipeline_imputes_rare_gaps_with_training_medians():
    rng = np.random.default_rng(2)
    y = np.repeat(["A", "B"], 30)
    X = np.where(y[:, None] == "A", 3.0, -3.0) + rng.normal(size=(60, 5))
    X_gappy = X.copy()
    X_gappy[0, 1] = np.nan
    model = logistic_pipeline(k=5, C=1.0).fit(X_gappy, y)
    assert np.isclose(model.named_steps["impute"].statistics_[1], np.nanmedian(X_gappy[:, 1]))
    test = X[:2].copy()
    test[1, 3] = np.nan
    assert model.predict(test).tolist() == ["A", "A"]
