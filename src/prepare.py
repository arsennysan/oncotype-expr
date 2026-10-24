"""Turn the raw TCGA files into an analysis-ready dataset.

Steps (each is a small function so it can be tested on its own):
  1. parse TCGA barcodes: patient, sample type, tissue source site;
  2. keep one primary solid tumour (code 01) per patient, labelled by cancer type;
  3. split patients into train / validation / test, stratified by cancer type;
  4. choose genes on the training set only: drop platform-dependent genes (missing in whole
     batches of samples) and genes that are never expressed. Rare remaining gaps are imputed
     inside the model pipeline.

Metastases (code 06) are kept aside for a later robustness check.

    python -m src.prepare
"""
from __future__ import annotations

import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

SEED = 42
PRIMARY, METASTASIS = "01", "06"


def parse_barcode(sample: str) -> dict:
    """TCGA-OR-A5J1-01 -> patient TCGA-OR-A5J1, site OR, sample type 01."""
    parts = sample.split("-")
    if len(parts) < 4 or parts[0] != "TCGA":
        raise ValueError(f"not a TCGA sample barcode: {sample!r}")
    return {"sample": sample, "patient": "-".join(parts[:3]), "site": parts[1], "sample_type": parts[3][:2]}


def build_samples(expr_samples: list[str], clinical: pd.DataFrame) -> pd.DataFrame:
    """One row per sample that has expression data and a known cancer type."""
    df = pd.DataFrame([parse_barcode(s) for s in expr_samples])
    labels = clinical.set_index("sample")["cancer type abbreviation"].dropna()
    df["cancer_type"] = df["sample"].map(labels)
    return df.dropna(subset=["cancer_type"]).reset_index(drop=True)


def select_samples(samples: pd.DataFrame, sample_type: str) -> pd.DataFrame:
    """Samples of one type, at most one per patient (the first barcode in sorted order)."""
    sel = samples[samples["sample_type"] == sample_type].sort_values("sample")
    return sel.drop_duplicates("patient", keep="first").reset_index(drop=True)


def split_patients(samples: pd.DataFrame, test_size: float = 0.15, val_size: float = 0.15, seed: int = SEED) -> pd.Series:
    """Assign each sample to train / val / test. Samples are one per patient, so no patient
    can appear in two sets; stratifying keeps every cancer type's share the same in each set."""
    idx = samples.index.to_numpy()
    y = samples["cancer_type"].to_numpy()
    rest, test = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    train, val = train_test_split(rest, test_size=val_size / (1 - test_size), stratify=y[rest], random_state=seed)
    split = pd.Series("train", index=samples.index)
    split[val] = "val"
    split[test] = "test"
    return split


def gene_symbol(row_id: str) -> str | None:
    """Row ids are gene symbols ('TP53'), or bare Entrez ids ('100130426') for unnamed genes.
    Some Xena tables use 'TP53|7157'; both forms are handled. Unnamed genes give None."""
    sym = row_id.split("|")[0]
    return None if sym in ("?", "") or sym.isdigit() else sym


def filter_genes(expr: pd.DataFrame, train_samples: list[str], max_missing: float = 0.01,
                 min_level: float = 1.0, min_share: float = 0.005) -> pd.DataFrame:
    """Choose genes using the training samples only, so validation and test data never
    influence the gene list.

    - Unnamed and duplicated genes are dropped.
    - Genes missing in more than `max_missing` of training tumours are dropped. In this dataset
      such gaps follow the sequencing platform: ~3,000 genes are absent for whole batches of six
      cancer types. Filling them in would let a model learn the platform instead of the biology.
      Rarer gaps (under 1%) are kept and imputed later, inside the model pipeline.
    - Genes must reach `min_level` (log2) in at least `min_share` of training tumours. A share,
      not a mean, keeps tissue-specific markers that are high in one small cancer type only.
    """
    symbols = expr.index.map(gene_symbol)
    named = pd.notna(symbols)
    expr = expr[named]
    expr.index = symbols[named]
    expr = expr[~expr.index.duplicated(keep="first")]
    train = expr[train_samples]
    missing_ok = train.isna().mean(axis=1) <= max_missing
    expressed = (train >= min_level).mean(axis=1) >= min_share
    return expr[missing_ok & expressed]


def read_expression(path: Path, wanted: set[str]) -> pd.DataFrame:
    """Read only the wanted sample columns, as float32 (the full table is ~1.8 GB as float64)."""
    with gzip.open(path, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
    cols = [header[0]] + [c for c in header[1:] if c in wanted]
    return pd.read_csv(path, sep="\t", index_col=0, usecols=cols, dtype={c: np.float32 for c in cols[1:]})


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    clinical = pd.read_csv(RAW / "clinical.tsv", sep="\t", usecols=["sample", "cancer type abbreviation"])
    with gzip.open(RAW / "expression.tsv.gz", "rt") as f:
        expr_samples = f.readline().rstrip("\n").split("\t")[1:]

    samples = build_samples(expr_samples, clinical)
    primary = select_samples(samples, PRIMARY)
    primary["split"] = split_patients(primary)
    metastases = select_samples(samples, METASTASIS)
    metastases["split"] = "metastasis"
    print(f"primary tumours: {len(primary)} samples, {primary['cancer_type'].nunique()} cancer types")
    print(f"metastases kept aside: {len(metastases)}")
    print(primary["split"].value_counts().to_string())

    wanted = set(primary["sample"]) | set(metastases["sample"])
    expr = read_expression(RAW / "expression.tsv.gz", wanted)
    print(f"expression read: {expr.shape[0]} genes x {expr.shape[1]} samples")
    train_ids = primary.loc[primary["split"] == "train", "sample"].tolist()
    expr = filter_genes(expr, train_ids)
    print(f"genes after filtering: {expr.shape[0]}, of which with rare gaps (imputed later): {int(expr.isna().any(axis=1).sum())}")

    meta = pd.concat([primary, metastases], ignore_index=True)
    meta.to_csv(OUT / "samples.csv", index=False)
    # Samples as rows, genes as columns: the layout scikit-learn expects.
    expr.T.loc[meta["sample"]].to_pickle(OUT / "expression.pkl")
    print(f"saved {OUT / 'samples.csv'} and {OUT / 'expression.pkl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
