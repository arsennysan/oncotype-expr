"""Tests for the external-validation helpers on small synthetic tables."""
import numpy as np
import pandas as pd

from src.external import collapse_to_symbols, ensembl_to_symbol, quantile_to_reference, reference_profile, score


def test_quantile_normalisation_keeps_ranks_and_takes_the_reference_scale():
    reference = reference_profile(np.array([[0.0, 2.0, 4.0, 6.0], [2.0, 4.0, 6.0, 8.0]]))
    assert reference.tolist() == [1.0, 3.0, 5.0, 7.0]
    # An external sample on a completely different scale (and in a different gene order).
    X = np.array([[-9.0, 100.0, -3.0, 20.0]])
    out = quantile_to_reference(X, reference)
    assert out.tolist() == [[1.0, 7.0, 3.0, 5.0]]


def test_quantile_normalisation_is_per_sample_and_leaves_gaps():
    reference = np.array([0.0, 1.0, 2.0])
    X = np.array([[5.0, 6.0, 7.0], [7.0, np.nan, 5.0], [1.0, 1.0, 9.0]])
    out = quantile_to_reference(X, reference)
    assert out[0].tolist() == [0.0, 1.0, 2.0]
    assert out[1, 0] == 2.0 and out[1, 2] == 0.0 and np.isnan(out[1, 1])
    assert out[2].tolist() == [0.5, 0.5, 2.0]  # tied genes share the reference values they span


def test_ensembl_ids_map_to_symbols_and_duplicates_collapse():
    probemap = pd.DataFrame({"id": ["ENSG01.5", "ENSG02.1", "ENSG03.2"], "gene": ["ALB", "TG", "TG"]})
    matrix = pd.DataFrame({"s1": [9.0, 1.0, 5.0, 3.0], "s2": [8.0, 2.0, 6.0, 3.0]},
                          index=["ENSG01.7", "ENSG02.1", "ENSG03.2", "ENSG99.1"])
    out = collapse_to_symbols(matrix, ensembl_to_symbol(matrix.index, probemap))
    assert sorted(out.columns) == ["ALB", "TG"]          # unmapped id dropped
    assert out.loc["s1", "ALB"] == 9.0                    # matched despite a different version
    assert out.loc["s1", "TG"] == 5.0                     # the higher-expressed duplicate wins


def test_score_accepts_any_allowed_type():
    classes = np.array(["COAD", "LUAD", "LUSC", "READ"])
    proba = np.array([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
    pred = classes[proba.argmax(axis=1)]
    out = score(pred, proba, classes, [{"COAD", "READ"}, {"LUSC"}])
    assert out == {"n": 2, "accuracy": 0.5, "top3_accuracy": 1.0}
