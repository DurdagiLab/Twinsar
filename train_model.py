"""
Train pIC50 Prediction Model
==============================

Train a machine learning model for pIC50 prediction using Morgan fingerprints
and molecular descriptors.

Usage:
    python train_model.py --data data/training_data.csv --output models/my_model.joblib

Requirements:
    - Training data with SMILES and pIC50 columns
    - RDKit for molecular descriptors
    - CatBoost (recommended) or scikit-learn for training
"""

import os
import gc
import json
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem, DataStructs

    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("ERROR: RDKit required. Install with: conda install -c conda-forge rdkit")
    exit(1)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import r2_score, mean_squared_error

MORGEN_RADIUS = 2
MORGEN_BITS = 128
DESCRIPTOR_BATCH_SIZE = 1000


def compute_descriptors(smi):
    """Compute molecular descriptors."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        return {
            "MolWt": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "TPSA": Descriptors.TPSA(mol),
            "NumHBA": Descriptors.NumHAcceptors(mol),
            "NumHBD": Descriptors.NumHDonors(mol),
            "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
            "NumAromaticRings": Descriptors.NumAromaticRings(mol),
            "NumRings": Descriptors.NumRings(mol),
            "FractionCSP3": Descriptors.FractionCSP3(mol),
            "NumHeavyAtoms": Descriptors.HeavyAtomCount(mol),
            "NumHeteroatoms": Descriptors.NumHeteroatoms(mol),
            "NumSaturatedRings": Descriptors.NumSaturatedRings(mol),
            "NumAliphaticRings": Descriptors.NumAliphaticRings(mol),
            "NumAromaticHeterocycles": Descriptors.NumAromaticHeterocycles(mol),
            "NumBridgeheadAtoms": Descriptors.NumBridgeheadAtoms(mol),
            "NumSpiroAtoms": Descriptors.NumSpiroAtoms(mol),
            "NumAmideBonds": Descriptors.NumAmideBonds(mol),
            "Kappa1": Descriptors.Kappa1(mol),
            "Kappa2": Descriptors.Kappa2(mol),
            "Kappa3": Descriptors.Kappa3(mol),
            "Chi0": Descriptors.Chi0(mol),
            "Chi1": Descriptors.Chi1(mol),
            "Chi0n": Descriptors.Chi0n(mol),
            "Chi1n": Descriptors.Chi1n(mol),
            "LabuteASA": Descriptors.LabuteASA(mol),
            "BalabanJ": Descriptors.BalabanJ(mol),
            "BertzCT": Descriptors.BertzCT(mol),
        }
    except:
        return None


def compute_morgan_fp(smi, radius=MORGEN_RADIUS, n_bits=MORGEN_BITS):
    """Compute Morgan fingerprint."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def load_and_prepare_data(file_path):
    """Load training data and compute features."""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)

    if "SMILES" not in df.columns:
        for col in df.columns:
            if "smiles" in col.lower():
                df.rename(columns={col: "SMILES"}, inplace=True)
                break

    if "pIC50" not in df.columns:
        for col in df.columns:
            if "pic50" in col.lower() or "activity" in col.lower():
                df.rename(columns={col: "pIC50"}, inplace=True)
                break

    if "SMILES" not in df.columns or "pIC50" not in df.columns:
        print("ERROR: Need SMILES and pIC50 columns")
        return None

    df = df.dropna(subset=["SMILES", "pIC50"])
    print(f"  Loaded {len(df)} molecules")

    print("  Computing descriptors...")
    desc_list = []
    for i, smi in enumerate(df["SMILES"]):
        d = compute_descriptors(smi)
        desc_list.append(d if d else {})
        if (i + 1) % DESCRIPTOR_BATCH_SIZE == 0:
            print(f"    {i + 1}/{len(df)}")
    df_desc = pd.DataFrame(desc_list)

    print("  Computing Morgan fingerprints...")
    fp_list = []
    for i, smi in enumerate(df["SMILES"]):
        fp = compute_morgan_fp(smi)
        fp_list.append(fp if fp is not None else np.zeros(MORGEN_BITS))
        if (i + 1) % DESCRIPTOR_BATCH_SIZE == 0:
            print(f"    {i + 1}/{len(df)}")

    morgan_cols = [f"morgan_{i}" for i in range(MORGEN_BITS)]
    df_morgan = pd.DataFrame(fp_list, columns=morgan_cols)

    X = pd.concat([df_desc.reset_index(drop=True), df_morgan], axis=1)
    y = df["pIC50"].values

    valid_rows = ~X.isnull().all(axis=1)
    X = X[valid_rows].reset_index(drop=True)
    y = y[valid_rows]

    print(f"  Valid molecules: {len(X)}")
    return X, y


def train_model(X, y, test_size=0.2, random_state=42):
    """Train the model."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"\nTraining set: {len(X_train)}, Test set: {len(X_test)}")

    print("  Imputing missing values...")
    imputer = KNNImputer(n_neighbors=5)
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    print("  Scaling features...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    print("  Training model...")

    try:
        import catboost as cb

        model = cb.CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=3,
            random_seed=random_state,
            verbose=100,
            early_stopping_rounds=50,
            task_type="CPU",
        )

        model.fit(
            X_train_scaled,
            y_train,
            eval_set=(X_test_scaled, y_test),
            verbose=100,
        )

        model_type = "CatBoost"

    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor

        print("  Using scikit-learn GradientBoosting (CatBoost not available)")
        model = GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            random_state=random_state,
            verbose=1,
        )
        model.fit(X_train_scaled, y_train)
        model_type = "GradientBoosting"

    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print(f"\n{'=' * 40}")
    print(f"Model: {model_type}")
    print(f"{'=' * 40}")
    print(f"Training R²:   {train_r2:.4f}")
    print(f"Test R²:       {test_r2:.4f}")
    print(f"Training RMSE: {train_rmse:.4f}")
    print(f"Test RMSE:     {test_rmse:.4f}")
    print(f"{'=' * 40}\n")

    return (
        model,
        scaler,
        imputer,
        {
            "train_r2": train_r2,
            "test_r2": test_r2,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "model_type": model_type,
            "n_features": X.shape[1],
            "n_train": len(X_train),
            "n_test": len(X_test),
        },
    )


def save_model(model, scaler, imputer, metrics, output_path, feature_names):
    """Save model to file."""
    data = {
        "model": model,
        "scaler": scaler,
        "imputer": imputer,
        "feature_names": feature_names,
        "metrics": metrics,
        "morgan_bits": MORGEN_BITS,
        "morgan_radius": MORGEN_RADIUS,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    import joblib

    joblib.dump(data, output_path)
    print(f"Model saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train pIC50 prediction model")
    parser.add_argument(
        "--data", required=True, help="Training data CSV (SMILES, pIC50)"
    )
    parser.add_argument("--output", required=True, help="Output model path (.joblib)")
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Test set fraction"
    )
    args = parser.parse_args()

    X, y = load_and_prepare_data(args.data)
    if X is None:
        return

    model, scaler, imputer, metrics = train_model(X, y, test_size=args.test_size)

    save_model(model, scaler, imputer, metrics, args.output, list(X.columns))


if __name__ == "__main__":
    main()
