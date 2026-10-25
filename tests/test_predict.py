"""Tests for the single-tumour report on a synthetic two-marker model."""
import numpy as np

from src.models import logistic_pipeline
from src.predict import describe, explain, top_guesses


def fitted_model():
    rng = np.random.default_rng(5)
    y = np.repeat(["LIHC", "THCA", "BRCA"], 40)
    X = rng.normal(size=(120, 5))
    X[y == "LIHC", 0] += 4   # ALB marks liver
    X[y == "THCA", 1] += 4   # TG marks thyroid
    genes = np.array(["ALB", "TG", "NOISE1", "NOISE2", "NOISE3"])
    return logistic_pipeline(k=5, C=1.0).fit(X, y), X, y, genes


def test_top_guesses_are_sorted_and_cut_to_k():
    out = top_guesses(np.array([0.1, 0.6, 0.05, 0.25]), np.array(["A", "B", "C", "D"]), k=3)
    assert out == [("B", 0.6), ("D", 0.25), ("A", 0.1)]


def test_explain_puts_the_marker_first_and_flags_it():
    model, X, y, genes = fitted_model()
    liver = X[np.argmax(np.where(y == "LIHC", X[:, 0], -np.inf))]   # the clearest liver sample
    table = explain(model, liver, genes, "LIHC", n=3)
    assert table.gene.iloc[0] == "ALB" and table.known_marker.iloc[0]
    assert table.push.is_monotonic_decreasing
    # z is the gene's standardised expression: clearly above average for a liver marker in a liver sample.
    assert table.z.iloc[0] > 1


def test_describe_reports_the_true_type_and_the_verdict():
    model, X, y, genes = fitted_model()
    i = np.argmax(np.where(y == "THCA", X[:, 1], -np.inf))         # the clearest thyroid sample
    text = describe("S1", "test", "THCA", model, X[i], genes, n_genes=3)
    assert "True type: THCA" in text and "RIGHT" in text and "<- correct" in text and "TG" in text
    # Claiming the wrong truth flips the verdict and adds an explanation for the claimed type.
    text = describe("S1", "test", "LIHC", model, X[i], genes, n_genes=3)
    assert "WRONG" in text and "towards LIHC" in text
