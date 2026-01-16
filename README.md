## Problem Statement

Glass transition temperature (Tg) is a critical thermal property of polymers that strongly influences their mechanical performance and application range.  
However, experimental measurement of Tg is time-consuming and resource-intensive, especially during early-stage material design.

The goal of this project is to build a data-driven machine learning pipeline that predicts polymer Tg directly from molecular structure.  
Given a polymer represented as a SMILES string, the system aims to estimate its Tg using molecular descriptors and an interpretable ML model.

This project focuses on building a clean, reproducible, end-to-end pipeline rather than optimizing for maximum predictive accuracy.

## Dataset

The dataset consists of polymer SMILES representations paired with experimentally measured glass transition temperatures (Tg).  
Data were collected from publicly available polymer property datasets and literature sources.

Before modeling, the dataset is cleaned to remove invalid SMILES entries, missing Tg values, and duplicate records.

## Exploratory Data Analysis (EDA)

This project began with an exploratory data analysis phase to understand data quality, descriptor characteristics, and to guide feature engineering decisions. (notebooks 01-04)

During EDA, RDKit molecular descriptors were generated from validated SMILES strings. Descriptor quality was assessed by examining missing-value ratios, variance, and inter-feature correlations across the full dataset.

Key exploratory findings include:

- A subset of descriptors exhibited extremely high missing-value ratios and were deemed uninformative.

- Median imputation was sufficient to stabilize descriptor distributions during prototyping.

- Many descriptors showed near-zero variance or high pairwise correlation, indicating redundancy.

- Reasonable thresholds were identified for feature reduction, including missing-value ratio, variance cutoff, and correlation cutoff.

All statistics in the EDA phase were computed on the full dataset and were used only to inform design decisions.
They were not used for final model training or evaluation.

A separate baseline feature pipeline was later constructed, where the same rules were applied using statistics fitted exclusively on the training set to prevent data leakage.

## Environment Strategy

This project intentionally uses two isolated Python environments,
each serving a distinct role in the workflow.

### 1. RDKit Descriptor Generation (Conda Environment)

RDKit is used exclusively for molecular descriptor generation.
Due to limited pip support for RDKit on Windows (especially for
newer Python versions), descriptor generation was performed in
a dedicated conda environment.

This step is treated as a preprocessing stage and executed only
when raw molecular data changes.

Output:
- `data/intermediate/tg_with_rdkit_descriptors.csv`

### 2. Machine Learning Pipeline (Python venv + pip)

All downstream machine learning steps are executed in a standard
Python virtual environment (`venv`) using pip-managed packages.

This includes:
- Train/test splitting
- Feature engineering pipelines
- Model training and evaluation

The ML pipeline does not depend on RDKit and operates solely on
precomputed descriptor tables, ensuring reproducibility and
deployment compatibility.

Dependencies for this stage are fully captured in `requirements.txt`.

### Rationale

This separation follows the principle of separation of concerns:

- Conda is used where complex native dependencies are unavoidable.
- pip + venv is used for reproducible, deployable ML workflows.

This design avoids over-coupling the ML pipeline to heavy
chemistry toolchains while keeping the full workflow transparent
and reproducible.

## Baseline Model (Frozen)

The baseline model consists of a Random Forest regressor trained on RDKit-based
molecular descriptors using a fixed feature engineering pipeline.

All preprocessing steps, including missing-value imputation, feature filtering,
and numerical stabilization, were fitted exclusively on the training set and
applied consistently to the test set. The resulting model achieved an RMSE of
approximately 40 K and an R² of 0.88 on held-out test data.

This baseline is frozen and serves as a reference point for subsequent model
development and comparison.
