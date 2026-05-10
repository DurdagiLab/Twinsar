# TwinSAR-8

Molecular twin detection using 8 single-element ratios with full ML pipeline for pIC50 prediction.

## Overview

TwinSAR-8 finds molecular twins by comparing element composition ratios between query and target molecules. It uses only 8 single-element ratios (C, N, O, F, S, Br, Cl, I) to identify structurally similar molecules, then optionally predicts pIC50 values for the identified twins using a trained ML model.

## Features

- **8-Element Ratio Fingerprints**: Simple, interpretable molecular representations
- **Block-based Twin Detection**: FAST and BULK modes for efficient processing
- **Deterministic Results**: Reproducible results with configurable random seed
- **Drug-likeness Filter**: Extended criteria (MW, LogP, HBD, HBA, TPSA, rotatable bonds)
- **pIC50 Prediction**: Optional ML model for activity prediction
- **High-activity Flagging**: Automatically flag molecules with predicted pIC50 > 6.5

## Installation

```bash
# Create environment
conda create -n twinsar8 python=3.10
conda activate twinsar8

# Install RDKit
conda install -c conda-forge rdkit

# Install dependencies
pip install numpy pandas scipy scikit-learn joblib matplotlib seaborn

# Optional: CatBoost for better predictions
pip install catboost optuna

# Optional: Flask for web interface
pip install flask
```

Or simply:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Train a pIC50 Model (Optional)

```bash
python train_model.py --data data/sample_query.csv --output models/my_model.joblib
```

Your training data should have `SMILES` and `pIC50` columns:
```csv
SMILES,pIC50
CCO,7.2
CC(=O)O,6.5
c1ccccc1,5.1
```

### 2. Run Twin Detection Pipeline

```bash
# Basic twin detection (FAST mode - default)
python app_8.py --query data/sample_query.csv --target data/sample_target.csv

# Bulk mode (more accurate, slower)
python app_8.py --query data/sample_query.csv --target data/sample_target.csv --mode bulk

# With pIC50 prediction
python app_8.py --query data/sample_query.csv --target data/sample_target.csv --model models/my_model.joblib
```

## Drug-likeness Filter

The pipeline applies these criteria to filter drug-like molecules:
- Molecular Weight: 100 - 1000 Da
- LogP: -2 to 10
- HBD (Hydrogen Bond Donors): ≤ 6
- HBA (Hydrogen Bond Acceptors): ≤ 15
- TPSA (Topological Polar Surface Area): ≤ 250 Å²
- Rotatable Bonds: ≤ 20

## Output

The pipeline creates:
- `all_twin_pairs.csv` - All twin pairs with similarity scores and z-scores
- `unique_twin_targets.csv` - Unique target molecules found as twins
- `druglike_twins.csv` - Drug-like molecules (filter applied)
- `high_activity_twins.csv` - High-activity hits with predicted pIC50 (if model provided)
- `pipeline_stats.json` - Summary statistics

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--query` | Query CSV file (required) | - |
| `--target` | Target CSV file (required) | - |
| `--output` | Output directory | Auto-generated |
| `--mode` | Detection mode (`fast` or `bulk`) | `fast` |
| `--z-threshold` | Z-score threshold | 2.576 (99%) |
| `--model` | pIC50 model path | None |
| `--no-predict` | Skip pIC50 prediction | False |
| `--high-activity-threshold` | High activity cutoff | 6.5 |

## Example

```bash
# Run with sample data (FAST mode)
python app_8.py --query data/sample_query.csv --target data/sample_target.csv

# Bulk mode for maximum accuracy
python app_8.py --query data/sample_query.csv --target data/sample_target.csv --mode bulk

# Custom threshold and model
python app_8.py \
    --query query.csv \
    --target target.csv \
    --model models/my_model.joblib \
    --z-threshold 3.0 \
    --high-activity-threshold 7.0
```

## Pre-trained Models

A pre-trained CatBoost model is available in:
```
models/legacy_default/final_catboost_regressor_model.joblib
```

With associated preprocessing components:
- `catboost_minmax_scaler.joblib`
- `median_imputer.joblib`
- `variance_threshold_selector.joblib`
- `selected_feature_names.json`

Usage:
```bash
python app_8.py \
    --query data/sample_query.csv \
    --target data/sample_target.csv \
    --model models/legacy_default/final_catboost_regressor_model.joblib
```

## Validation

To validate the twin detection algorithm:
```bash
python validate_8elem_filtered_v2.py
```

This runs BULK and FAST modes and produces comparison plots including:
- Ratio similarity analysis
- Negative control tests
- Gamma distribution analysis
- Z-score discrimination

## How It Works

### 1. Ratio Calculation
```
ratio_X = count(X) / total_heavy_atoms
```
For each molecule, compute 8 ratios (C, N, O, F, S, Br, Cl, I).

### 2. pIC50 Pre-filter
Query molecules with known pIC50 values are filtered (pIC50 > 6.5) to use only high-activity compounds as twin search anchors.

### 3. Twin Detection (Block-based)
1. Group molecules by fingerprint block (binary pattern of present/absent elements)
2. Compute RBF kernel similarity only within matching blocks
3. Apply logit transform to similarities
4. Calculate z-scores based on target distribution
5. Flag molecules with z-score > threshold as twins

### 4. Drug-likeness Filter
Apply extended drug-likeness criteria to filter twin targets.

### 5. pIC50 Prediction (Optional)
If a trained model is provided, predict pIC50 for drug-like twins and flag high-activity hits (pIC50 > threshold).

## Project Structure

```
twinsar-8/
├── app_8.py                    # Main application (CLI)
├── train_model.py              # Model training script
├── validate_8elem_filtered_v2.py  # Validation script
├── requirements.txt            # Dependencies
├── README.md                   # This file
├── data/                       # Sample data
│   ├── sample_query.csv
│   └── sample_target.csv
├── query_files/                # Query molecules directory
├── target_files/               # Target molecules directory
└── models/                     # Trained models
    └── legacy_default/         # Pre-trained model
        ├── final_catboost_regressor_model.joblib
        ├── catboost_minmax_scaler.joblib
        ├── median_imputer.joblib
        ├── variance_threshold_selector.joblib
        └── selected_feature_names.json
```

## License

MIT License