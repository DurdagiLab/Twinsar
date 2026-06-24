TwinSAR-8 - Usage Guide
========================

TwinSAR-8 detects molecular twins between query and target datasets using
8 single-element atom ratios (C, N, O, F, S, Br, Cl, I) computed from SMILES.

Prerequisites
-------------
- Python 3 with RDKit, pandas, numpy, scikit-learn
- catboost (optional, for pIC50 prediction)

Required Arguments
------------------
  --query   PATH   Path to query CSV file (must contain 'smiles' column)
  --target  PATH   Path to target CSV file (must contain 'smiles' column)

Optional Arguments
------------------
  --output  DIR    Output directory for results (default: creates timestamped dir)

  --mode   {fast,bulk}
                  Twin detection mode (default: fast)
                  - fast: optimized vectorized similarity (faster, slightly approximate)
                  - bulk: exact element-wise comparison (slower, exact)

  --model  PATH   Path to pIC50 prediction model (.joblib file)
                  If provided, predicts pIC50 for target molecules and flags
                  high-activity query twins matching those predictions.

  --z-threshold  FLOAT
                  Z-score threshold for twin significance (default: 2.576)
                  Higher values = stricter, fewer twins.

  --high-activity-threshold  FLOAT
                  pIC50 threshold for high-activity flag (default: 6.5)

  --no-predict   Skip pIC50 prediction even if a model is provided.

Examples
--------

  # Minimal run (no pIC50 prediction)
  python app_8.py --query data/sample_query.csv --target data/sample_target.csv

  # Bulk mode with legacy default model and pIC50 prediction
  python app_8.py --query data/sample_query.csv --target data/sample_target.csv \
      --mode bulk --model models/legacy_default/final_catboost_regressor_model.joblib \
      --output results

  # With custom z-score threshold, skip prediction
  python app_8.py --query data/sample_query.csv --target data/sample_target.csv \
      --z-threshold 3.0 --no-predict
