"""Tests for the normal-tissue background helpers."""
import numpy as np
import pandas as pd

from src.background import NORMAL, TumourOnly, pick_gtex_samples
from src.models import logistic_pipeline


def test_tumour_only_drops_normal_classes_and_renormalises():
    rng = np.random.default_rng(4)
    y = np.repeat(["BRCA", "LIHC", NORMAL + "Liver"], 30)
    X = rng.normal(size=(90, 4))
    X[y == "BRCA", 0] += 4
    X[y == "LIHC", 1] += 4
    X[y == NORMAL + "Liver", 2] += 4
    full = logistic_pipeline(k=4, C=1.0).fit(X, y)
    model = TumourOnly(full)
    assert model.classes_.tolist() == ["BRCA", "LIHC"]
    p = model.predict_proba(X)
    assert p.shape == (90, 2) and np.allclose(p.sum(axis=1), 1)
    # Ratios between tumour classes are unchanged by dropping the normal class.
    raw = full.predict_proba(X)
    assert np.allclose(p[:, 0] / p[:, 1], raw[:, 0] / raw[:, 1])
    assert set(model.predict(X)) <= {"BRCA", "LIHC"}


def test_gtex_sampling_caps_tissues_skips_cell_lines_and_rare_tissues():
    rows = []
    for i in range(60):
        rows.append({"Sample": f"L{i}", "_patient": f"D{i}", "_primary_site": "Liver", "body_site_detail (SMTSD)": "Liver"})
        rows.append({"Sample": f"F{i}", "_patient": f"D{i}", "_primary_site": "Skin",
                     "body_site_detail (SMTSD)": "Cells - Cultured fibroblasts"})
    for i in range(10):
        rows.append({"Sample": f"B{i}", "_patient": f"D{i}", "_primary_site": "Bladder", "body_site_detail (SMTSD)": "Bladder"})
    rows.append({"Sample": "L_dup", "_patient": "D0", "_primary_site": "Liver", "body_site_detail (SMTSD)": "Liver"})
    picked = pick_gtex_samples(pd.DataFrame(rows), per_tissue=40)
    assert picked.label.unique().tolist() == [NORMAL + "Liver"]   # fibroblasts and rare bladder gone
    assert len(picked) == 40                                       # capped
    assert not picked.duplicated(["_patient", "_primary_site"]).any()
