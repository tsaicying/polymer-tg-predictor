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