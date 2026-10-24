"""Tests for data preparation on small synthetic tables: barcode parsing, sample selection,
leakage-free splitting and training-only gene filtering."""
import numpy as np
import pandas as pd
import pytest

from src.prepare import build_samples, filter_genes, parse_barcode, select_samples, split_patients


def test_parse_barcode():
    assert parse_barcode("TCGA-OR-A5J1-01") == {"sample": "TCGA-OR-A5J1-01", "patient": "TCGA-OR-A5J1", "site": "OR", "sample_type": "01"}
    assert parse_barcode("TCGA-D3-A1QA-07A")["sample_type"] == "07"
    with pytest.raises(ValueError):
        parse_barcode("GTEX-1117F-0226")


def test_build_samples_drops_unlabelled():
    clinical = pd.DataFrame({"sample": ["TCGA-AA-0001-01", "TCGA-AA-0002-01"], "cancer type abbreviation": ["COAD", None]})
    out = build_samples(["TCGA-AA-0001-01", "TCGA-AA-0002-01", "TCGA-AA-0003-01"], clinical)
    assert out["sample"].tolist() == ["TCGA-AA-0001-01"]
    assert out["cancer_type"].tolist() == ["COAD"]


def test_select_samples_one_primary_per_patient():
    samples = pd.DataFrame([
        parse_barcode(s) | {"cancer_type": "LUAD"}
        for s in ["TCGA-05-0001-01", "TCGA-05-0001-11", "TCGA-05-0002-01", "TCGA-05-0002-06", "TCGA-05-0003-06"]
    ])
    primary = select_samples(samples, "01")
    assert primary["sample"].tolist() == ["TCGA-05-0001-01", "TCGA-05-0002-01"]
    assert select_samples(samples, "06")["patient"].tolist() == ["TCGA-05-0002", "TCGA-05-0003"]


def make_cohort(n_per_type=(200, 120, 40)):
    rows = []
    for t, n in zip(["BRCA", "LUAD", "CHOL"], n_per_type):
        rows += [{"sample": f"TCGA-XX-{t}{i:04d}-01", "patient": f"TCGA-XX-{t}{i:04d}", "cancer_type": t} for i in range(n)]
    return pd.DataFrame(rows)


def test_split_has_no_patient_in_two_sets_and_is_stratified():
    cohort = make_cohort()
    split = split_patients(cohort)
    assert set(split) == {"train", "val", "test"}
    per_patient = cohort.assign(split=split).groupby("patient")["split"].nunique()
    assert (per_patient == 1).all()
    shares = pd.crosstab(cohort["cancer_type"], split, normalize="index")
    assert np.allclose(shares["test"], 0.15, atol=0.03)
    assert np.allclose(shares["train"], 0.70, atol=0.03)
    # Even the smallest class reaches the test set.
    assert (cohort.assign(split=split).query("cancer_type == 'CHOL'")["split"] == "test").sum() >= 5


def test_split_is_reproducible():
    cohort = make_cohort()
    assert split_patients(cohort).equals(split_patients(cohort))


def test_gene_symbol_handles_both_id_formats():
    from src.prepare import gene_symbol
    assert gene_symbol("TP53") == "TP53"
    assert gene_symbol("TP53|7157") == "TP53"
    assert gene_symbol("NKX2-1") == "NKX2-1"
    assert gene_symbol("100130426") is None
    assert gene_symbol("?|100130426") is None


def test_filter_genes_uses_training_samples_only():
    train = [f"tr{i}" for i in range(200)]
    expr = pd.DataFrame(0.0, index=["TP53", "XIST", "100130426", "KLK3", "PLATFORM", "TP53"], columns=train + ["test1"])
    expr.loc["TP53", train] = 5.0
    expr.loc["XIST", "test1"] = 9.0            # high only in the test sample
    expr.loc["100130426", train] = 4.0         # unnamed gene
    expr.loc["KLK3", train[:10]] = 15.0        # high in 5% of training tumours only (a tissue marker)
    expr.loc["KLK3", train[10]] = np.nan       # one rare gap: kept, imputed later
    expr.loc["PLATFORM", train] = 3.0
    expr.loc["PLATFORM", train[:20]] = np.nan  # missing in 10%: a platform-dependent gene
    out = filter_genes(expr, train)
    assert out.index.tolist() == ["TP53", "KLK3"]
    assert out.loc["KLK3"].isna().sum() == 1
