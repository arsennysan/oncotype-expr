# OncoType-Expr

[![tests](https://github.com/arsennysan/oncotype-expr/actions/workflows/tests.yml/badge.svg)](https://github.com/arsennysan/oncotype-expr/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Which tissue did a tumour come from? A pan-cancer classifier on TCGA RNA-seq, built to be leakage-safe and interpretable.**

About 3–5% of cancer patients present with metastases whose primary site cannot be found ("cancer of unknown primary"). Gene-expression classifiers are one way to suggest the tissue of origin. This project builds such a classifier on public TCGA data and focuses on doing it honestly: no information from the test set leaks into training, the metrics hold up under class imbalance, and the errors are explained biologically.

> **Status:** stages 0–5 done: data, exploration, models, SHAP interpretation, robustness on metastases, unseen hospitals and small gene panels. See [Roadmap](#roadmap).

**Highlights**

- Test macro-F1 **0.956**, accuracy **97.0%**, right answer in the top three for **99.8%** of tumours (32 cancer types).
- The model's top genes are textbook tissue markers: STAR, MLANA, KLK2, TG and DBH rank **#1** for their tissue; ALB, WT1 and SFTPB rank #2.
- **Metastases** it never saw: 92% accuracy, 98% top-3. **Hospitals** it never saw: macro-F1 drops only from 0.938 to 0.916.
- **50 genes** give over 90% of the full performance; 500 genes match it.
- Found and fixed a data trap: ~3,000 genes are missing only in certain sequencing batches, so imputing them would let the model recognise the batch instead of the tumour.

## Results

Test set: 1,452 primary tumours, 32 cancer types, scored once after model choice on the validation set.

| Model | Macro-F1 | Balanced accuracy | Accuracy | Top-3 accuracy |
|---|---|---|---|---|
| Majority class (reference) | 0.006 | 0.031 | 0.113 | 0.127 |
| **Logistic regression** (L2, C = 0.1, 5,000 genes), chosen on validation | **0.956** | **0.958** | **0.970** | **0.998** |
| LightGBM (2,000 genes) | 0.930 | 0.934 | 0.957 | 0.992 |

**Where the model errs.** 44 of 1,452 test tumours are misclassified, and the errors concentrate in biologically close pairs:

| Confusion | Errors | Why it is expected |
|---|---|---|
| Colon (COAD) ↔ rectum (READ) | 19 | Essentially one disease, colorectal adenocarcinoma, split by anatomical site |
| Oesophagus (ESCA) ↔ stomach (STAD) | 4 | Oesophageal adenocarcinoma near the junction is molecularly close to gastric cancer |
| Low-grade glioma (LGG) ↔ glioblastoma (GBM) | 2 | A grading continuum of the same tissue |
| Kidney clear-cell (KIRC) ↔ papillary (KIRP) | 2 | Two subtypes of renal cell carcinoma |
| Cervix (CESC) ↔ endometrium (UCEC) | 2 | Neighbouring parts of the uterus |

The right answer is among the top three guesses for 99.8% of tumours.

![Confusion matrix](results/figures/confusion_matrix.png)

## What the exploration showed

![UMAP of TCGA tumours](results/figures/umap_cancer_types.png)

- Most cancer types form their own clusters (87% of a tumour's 15 nearest neighbours share its type). Organs with a shared origin sit together: kidney subtypes, liver and bile ducts, the digestive tract, and **squamous cancers of different organs** (head and neck, lung, cervix, oesophagus), where cell type matters more than organ.
- Breast cancer splits into two clouds; the smaller one is the basal-like ("triple-negative") subtype.
- **Collection-site signal:** within breast cancer, the hospital that collected a sample can be guessed from expression at about twice the chance rate (15% vs 7%). The signal is weak next to the differences between cancer types, but it is real, so a held-out-site test is on the roadmap.

Details and explanations (in Russian) are in [`notebooks/01_explore.ipynb`](notebooks/01_explore.ipynb).

## What the model looks at (SHAP)

For logistic regression, a gene's SHAP value is exact: coefficient × standardised expression (checked against `shap.LinearExplainer`, difference 2·10⁻⁸). Averaging over each cancer type's test tumours gives its most important genes. The list of known markers was written down **before** looking at the results.

![Top genes per cancer type](results/figures/shap_top_genes.png)

| Cancer type | Known marker | Rank among 5,000 genes |
|---|---|---|
| Adrenocortical (ACC) | STAR | 1 |
| Melanoma, skin and eye (SKCM, UVM) | MLANA | 1 |
| Prostate (PRAD) | KLK2 | 1 |
| Thyroid (THCA) | TG | 1 |
| Pheochromocytoma (PCPG) | DBH | 1 |
| Liver (LIHC) | ALB | 2 |
| Ovary (OV) | WT1 | 2 |
| Lung adenocarcinoma (LUAD) | SFTPB | 2 |
| Bladder (BLCA), mesothelioma (MESO), pancreas (PAAD) | UPK2, UPK3B, CPA2 | 3 |
| Testis (TGCT) | POU5F1 | 5 |

Of 55 markers present in the gene set, 27 are in the top 10 for their tissue and 42 in the top 50 ([full table](results/marker_check.csv)). Two findings stand out:

- **Shared lineage markers rank low.** PAX8 (kidney, ovary, endometrium) and KRT5 (all squamous cancers) are real markers but do not separate classes, so the model relies on other genes.
- **The model partly reads the surrounding normal tissue.** For squamous lung cancer (LUSC) the top genes are surfactants SFTPA1/SFTPA2/SFTPB, which come from normal lung alveoli, not tumour cells. This works for primary tumours and predicts trouble for metastases, which is what the robustness check found.

## Robustness

The test set was used once and is not touched here; checks 2 and 3 use the training and validation tumours only (`src/robustness.py`).

**1. Metastases** (394 TCGA metastases, never seen in training; the model learned on primary tumours only).

| | n | Accuracy |
|---|---|---|
| All metastases | 394 | 0.924 (top-3: 0.980) |
| Melanoma | 368 | 0.932 |
| Other types | 26 | 0.808 |

Melanoma errors are mostly sarcoma (12; spindle-cell melanomas can look like sarcoma even under the microscope) and uveal melanoma (4, the same melanocyte lineage). Two breast cancer metastases were called liver cancer and mesothelioma, most likely because they were sampled from the liver and pleura: the same normal-tissue effect seen in the SHAP analysis.

**2. Hospitals the model never saw.** Five-fold cross-validation on 8,224 tumours from 680 collection sites, split at random vs. grouped by site (`GroupKFold`), so every held-out tumour comes from an unseen hospital.

| Split | Macro-F1 | Accuracy |
|---|---|---|
| Random | 0.938 | 0.957 |
| Unseen sites | 0.916 | 0.942 |

The drop is small: the model uses a little site-specific signal, but its main signal transfers to new centres.

**3. How many genes are enough?** Genes ranked by model weight, model refitted on the top k, scored on validation.

![Gene budget](results/figures/gene_budget.png)

50 genes reach macro-F1 0.909 (over 90% of the best), 500 genes reach the maximum of 0.946, and more genes add only noise. A targeted panel of a few hundred genes would be enough in principle.

Explanations in Russian: [`notebooks/02_interpret_and_robustness.ipynb`](notebooks/02_interpret_and_robustness.ipynb).

## A data trap: platform-dependent missing genes

The first version dropped every gene with any missing value and lost about 4,000 genes, including well-known markers. A closer look showed two different groups:

- **~3,000 genes are missing in 5–20% of samples, all from a few batches** (stomach, endometrium, ovary, oesophagus, colon, rectum) measured on a different platform version. Filling these gaps would let the model learn "value missing → stomach", i.e. recognise the batch instead of the tumour. They are excluded. The cost: lung markers NKX2-1 and SFTPC, glioma marker OLIG2 and melanoma marker TYR are lost.
- **~1,100 genes have rare gaps** (under 1% of samples, 27 stomach tumours). They are kept and filled with the training median inside the model pipeline.

## How leakage is avoided

- **One primary tumour per patient**, so no patient can be in both training and test data.
- **Stratified 70/15/15 split** of patients; every cancer type, even the smallest (36 tumours), reaches the test set.
- **Gene filtering uses training samples only** (`src/prepare.py`), and gene selection and scaling live inside scikit-learn pipelines, so they are fitted on training data and only applied to validation and test data.
- **Model choice on the validation set; the test set is scored once**, at the end.
- These rules are covered by unit tests (`tests/`).

## Data

[TCGA Pan-Cancer Atlas](https://gdc.cancer.gov/about-data/publications/pancanatlas) via [UCSC Xena](https://xenabrowser.net/datapages/?cohort=TCGA%20PanCanAtlas): batch-corrected RNA-seq (EB++, log2), sample annotations and the TCGA Clinical Data Resource (Liu et al., *Cell* 2018).

- 9,676 primary solid tumours from 32 cancer types; 17,456 genes after filtering.
- 394 metastases are kept aside for the robustness test.
- Acute myeloid leukaemia (LAML) is not included: it is a blood cancer with no solid primary tumour sample.

Data is downloaded by a script and never committed.

## Run it

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows; on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m src.download          # ~335 MB from UCSC Xena, resumable
python -m src.prepare           # ~1 min
python -m src.train             # ~2–3 min
python -m src.interpret         # SHAP top genes and marker check
python -m src.robustness        # metastases, unseen sites, gene budget (~15 min)
python -m pytest -q tests
```

## Project structure

```
src/download.py     fetch TCGA files (resumes dropped downloads, checks sizes)
src/prepare.py      barcodes, sample selection, leakage-free split, gene filtering
src/models.py       pipelines (top-variance genes → scaling → classifier) and metrics
src/train.py        model comparison, single test evaluation, figures
src/interpret.py    per-gene contributions (exact SHAP), known-marker check
src/robustness.py   metastases, unseen collection sites, gene budget
notebooks/          exploration with explanations
results/            metrics.json, per-class report, figures
tests/              unit tests on synthetic data, run by GitHub Actions
```

## Roadmap

- [x] Data download and leakage-free preparation
- [x] Exploration: class balance, UMAP, collection-site signal
- [x] Baseline models: logistic regression, LightGBM; honest test evaluation
- [x] SHAP interpretation: does the model recover known tissue markers?
- [x] Robustness: TCGA metastases, held-out collection sites
- [x] How few genes are enough for 90% of the performance
- [ ] External validation on independent data (other cohorts, CCLE cell lines)
- [ ] Small demo app

## Limitations

This is a learning and research project, **not a diagnostic tool**. TCGA tumours are high-quality research samples; real clinical samples are noisier, and accuracy on metastases is lower than on primary tumours (92% vs 97%), and only 26 non-melanoma metastases were available.

## Credits

Built with the help of [Claude Code](https://claude.com/claude-code).

## License

[MIT](LICENSE)
