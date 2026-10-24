"""Tests for the interpretation helpers on a synthetic two-gene signal."""
import numpy as np

from src.interpret import class_contributions, marker_check, top_genes_per_type
from src.models import logistic_pipeline


def fitted_model():
    rng = np.random.default_rng(3)
    y = np.repeat(["LIVER", "THYROID", "OTHER"], 40)
    X = rng.normal(size=(120, 6))
    X[y == "LIVER", 0] += 4    # gene 0 marks liver
    X[y == "THYROID", 2] += 4  # gene 2 marks thyroid
    genes = np.array(["ALB", "NOISE1", "TG", "NOISE2", "NOISE3", "NOISE4"])
    return logistic_pipeline(k=6, C=1.0).fit(X, y), X, y, genes


def test_contributions_add_up_to_the_class_scores():
    model, X, y, genes = fitted_model()
    contrib, names = class_contributions(model, X, genes)
    clf = model.named_steps["clf"]
    # Sum of contributions + intercept = the model's decision score for every class.
    scores = contrib.sum(axis=1) + clf.intercept_
    assert np.allclose(scores, model.decision_function(X), atol=1e-5)
    assert names.tolist() == genes.tolist()


def test_marker_genes_come_first_for_their_type():
    model, X, y, genes = fitted_model()
    contrib, names = class_contributions(model, X, genes)
    classes = model.named_steps["clf"].classes_
    top = top_genes_per_type(contrib, names, y, classes, n=2)
    assert top.query("cancer_type == 'LIVER' and rank == 1").gene.item() == "ALB"
    assert top.query("cancer_type == 'THYROID' and rank == 1").gene.item() == "TG"
    check = marker_check(contrib, names, y, classes, markers={"LIVER": ["ALB", "MISSING"], "THYROID": ["TG"]})
    assert check.set_index("marker").loc["ALB", "rank"] == 1
    assert check.set_index("marker").loc["TG", "rank"] == 1
    assert not check.set_index("marker").loc["MISSING", "in_model"]
