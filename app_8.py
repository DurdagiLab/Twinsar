"""
TwinSAR-8 - Twin Detection using 8 Single-Element Ratios
=========================================================

A local version of TwinSAR that uses only 8 single-element ratios
(C, N, O, F, S, Br, Cl, I) for molecular twin detection with full ML pipeline.

Usage:
    python app_8.py --query data/query.csv --target data/target.csv

    # With pIC50 prediction
    python app_8.py --query data/query.csv --target data/target.csv --model models/my_model.joblib

Dependencies:
    - RDKit for molecular operations
    - scikit-learn for similarity calculations
    - catboost for pIC50 prediction (optional)

Author: TwinSAR Team
License: MIT
"""

import os
import gc
import json
import time
import argparse
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem, DataStructs

    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print(
        "WARNING: RDKit not installed. Install with: conda install -c conda-forge rdkit"
    )

from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer

import joblib

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_ELEMENTS = ["C", "N", "O", "F", "S", "Br", "Cl", "I"]
Z_THRESHOLD = 2.576
EPSILON = 1e-7
DESCRIPTOR_BATCH_SIZE = 1000
MAX_GAMMA_SAMPLES = 5000
CHUNK_SIZE = 1000
CSV_CHUNK_SIZE = 50000
MORGAN_RADIUS = 2
MORGAN_BITS = 128
GLOBAL_SEED = 42

import random

random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)


def _rng(seed=GLOBAL_SEED):
    return np.random.RandomState(seed)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QUERY_DIR = os.path.join(BASE_DIR, "query_files")
TARGET_DIR = os.path.join(BASE_DIR, "target_files")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
MODELS_DIR = os.path.join(BASE_DIR, "models")

for _d in [QUERY_DIR, TARGET_DIR, UPLOAD_DIR, MODELS_DIR]:
    os.makedirs(_d, exist_ok=True)

# ============================================================================
# Core Descriptors for pIC50 Prediction
# ============================================================================

CORE_DESCRIPTORS = [
    "MolWt",
    "LogP",
    "TPSA",
    "NumHBA",
    "NumHBD",
    "NumRotatableBonds",
    "NumAromaticRings",
    "NumRings",
    "FractionCSP3",
    "NumHeavyAtoms",
    "NumHeteroatoms",
    "NumSaturatedRings",
    "NumAliphaticRings",
    "NumAromaticHeterocycles",
    "NumBridgeheadAtoms",
    "NumSpiroAtoms",
    "NumAmideBonds",
    "Kappa1",
    "Kappa2",
    "Kappa3",
    "Chi0",
    "Chi1",
    "Chi0n",
    "Chi1n",
    "LabuteASA",
    "BalabanJ",
    "BertzCT",
]


def compute_morgan_fp(smiles, radius=MORGAN_RADIUS, n_bits=MORGAN_BITS):
    """Compute Morgan fingerprint for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def compute_full_descriptors(smi):
    """Compute all descriptors needed for pIC50 prediction."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        desc = {
            "MolWt": Descriptors.MolWt(mol),
            "NumAtoms": mol.GetNumAtoms(),
            "NumHeavyAtoms": Descriptors.HeavyAtomCount(mol),
            "NumAromaticCarbocycles": Descriptors.NumAromaticCarbocycles(mol),
            "MolLogP": Descriptors.MolLogP(mol),
            "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
            "NumHDonors": Descriptors.NumHDonors(mol),
            "NumHAcceptors": Descriptors.NumHAcceptors(mol),
            "FractionCSP3": Descriptors.FractionCSP3(mol),
            "NumLipinskiHBA": Descriptors.NumHAcceptors(mol),
            "BertzCT": Descriptors.BertzCT(mol),
            "BalabanJ": Descriptors.BalabanJ(mol),
            "Kappa3": Descriptors.Kappa3(mol),
            "LabuteASA": Descriptors.LabuteASA(mol),
            "PEOE_VSA14": Descriptors.PEOE_VSA14(mol),
            "SMR_VSA10": Descriptors.SMR_VSA10(mol),
            "EState_VSA6": Descriptors.EState_VSA6(mol),
            "NumSaturatedRings": Descriptors.NumSaturatedRings(mol),
            "NumAliphaticRings": Descriptors.NumAliphaticRings(mol),
            "NumHeterocycles": Descriptors.NumHeterocycles(mol),
            "NumAliphaticHeterocycles": Descriptors.NumAliphaticHeterocycles(mol),
            "NumSpiroAtoms": Descriptors.NumSpiroAtoms(mol),
            "NumAmideBonds": Descriptors.NumAmideBonds(mol),
            "MaxAbsEStateIndex": Descriptors.MaxAbsEStateIndex(mol),
            "VSA_EState9": Descriptors.VSA_EState9(mol),
        }
        return desc
    except Exception:
        return None


def compute_descriptors_batch(smiles_list, batch_size=DESCRIPTOR_BATCH_SIZE):
    """Compute descriptors for a list of SMILES with progress."""
    n = len(smiles_list)
    desc_list = []
    for i, smi in enumerate(smiles_list):
        d = compute_full_descriptors(smi)
        desc_list.append(d if d else {})
        if (i + 1) % batch_size == 0 or i == n - 1:
            print(f"    Descriptors: {i + 1}/{n}")
    return pd.DataFrame(desc_list)


def compute_morgan_batch(smiles_list, batch_size=DESCRIPTOR_BATCH_SIZE):
    """Compute Morgan fingerprints for a list of SMILES."""
    n = len(smiles_list)
    fp_list = []
    for i, smi in enumerate(smiles_list):
        fp = compute_morgan_fp(smi)
        fp_list.append(fp if fp is not None else np.zeros(MORGAN_BITS))
        if (i + 1) % batch_size == 0 or i == n - 1:
            print(f"    Morgan FP: {i + 1}/{n}")
    return np.array(fp_list)


# ============================================================================
# pIC50 Prediction Model
# ============================================================================


class PIC50Predictor:
    """pIC50 prediction model wrapper with legacy model support."""

    def __init__(self, model_path=None):
        self.model = None
        self.scaler = None
        self.imputer = None
        self.variance_threshold = None
        self.feature_names = None
        self._loaded = False

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def load(self, model_path):
        """Load model from file (supports both bundled and legacy formats)."""
        if not os.path.exists(model_path):
            print(f"WARNING: Model file not found: {model_path}")
            return False

        try:
            data = joblib.load(model_path)
            model_dir = os.path.dirname(model_path)

            if isinstance(data, dict):
                self.model = data.get("model")
                self.scaler = data.get("scaler")
                self.imputer = data.get("imputer")
                self.feature_names = data.get("feature_names")
            else:
                self.model = data

                scaler_path = os.path.join(model_dir, "catboost_minmax_scaler.joblib")
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)

                imputer_path = os.path.join(model_dir, "median_imputer.joblib")
                if os.path.exists(imputer_path):
                    self.imputer = joblib.load(imputer_path)

                variance_path = os.path.join(
                    model_dir, "variance_threshold_selector.joblib"
                )
                if os.path.exists(variance_path):
                    self.variance_threshold = joblib.load(variance_path)

                features_path = os.path.join(model_dir, "selected_feature_names.json")
                if os.path.exists(features_path):
                    with open(features_path) as f:
                        self.feature_names = json.load(f)

            self._loaded = True
            print(f"  Loaded model from: {model_path}")
            if self.feature_names:
                print(f"  Using {len(self.feature_names)} features")
            return True
        except Exception as e:
            print(f"WARNING: Failed to load model: {e}")
            return False

    def predict(self, smiles_list, batch_size=DESCRIPTOR_BATCH_SIZE):
        """Predict pIC50 for a list of SMILES."""
        if not self._loaded or self.model is None:
            print("  WARNING: No model loaded, skipping pIC50 prediction")
            return None

        print(f"  Computing descriptors for {len(smiles_list)} molecules...")
        df_desc = compute_descriptors_batch(smiles_list, batch_size)

        print(f"  Computing Morgan fingerprints...")
        arr_morgan = compute_morgan_batch(smiles_list, batch_size)

        morgan_cols = [f"morgan_{i}" for i in range(MORGAN_BITS)]
        df_morgan = pd.DataFrame(arr_morgan, columns=morgan_cols)

        X = pd.concat([df_desc.reset_index(drop=True), df_morgan], axis=1)

        if self.feature_names:
            for col in self.feature_names:
                if col not in X.columns:
                    X[col] = 0
            X = X[self.feature_names]
        else:
            self.feature_names = list(X.columns)

        if self.variance_threshold is not None:
            X = self.variance_threshold.transform(X)

        if self.imputer:
            X = self.imputer.transform(X)

        if self.scaler:
            X = self.scaler.transform(X)

        try:
            predictions = self.model.predict(X)
            return predictions
        except Exception as e:
            print(f"  ERROR during prediction: {e}")
            return None


def get_model_for_prediction(model_path=None):
    """Get model for prediction."""
    if model_path is None:
        default_model = os.path.join(MODELS_DIR, "active_model.joblib")
        if os.path.exists(default_model):
            model_path = default_model

    if model_path is None or not os.path.exists(model_path):
        return None

    return PIC50Predictor(model_path)


# ============================================================================
# Ratio Computation (8-element mode)
# ============================================================================


def _generate_all_ratio_names(elements=None):
    """Generate only the 8 single-element ratio column names."""
    if elements is None:
        elements = DEFAULT_ELEMENTS
    return [f"ratio_{e}" for e in elements]


def _calc_ratios_for_smiles(smiles, elements):
    """Calculate only 8 single-element ratios for a single SMILES string."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    counts = {}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym != "H":
            counts[sym] = counts.get(sym, 0) + 1
    total = sum(counts.values())
    if total == 0:
        return None
    out = {}
    for e in elements:
        key = f"ratio_{e}"
        out[key] = counts.get(e, 0) / total
    return out


def _compute_ratios_batched(
    df, dataset_name, elements, batch_size=DESCRIPTOR_BATCH_SIZE
):
    """Compute element ratios for all molecules in a DataFrame."""
    n = len(df)
    all_nan = {k: np.nan for k in _generate_all_ratio_names(elements)}
    parts = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        rows = []
        for smi in df["SMILES"].iloc[start:end]:
            r = _calc_ratios_for_smiles(smi, elements)
            rows.append(r if r is not None else dict(all_nan))
        parts.append(pd.DataFrame(rows))
        if end % 5000 == 0 or end == n:
            print(f"  {dataset_name}: ratios {end}/{n}")
        gc.collect()
    ratio_df = pd.concat(parts, ignore_index=True)
    df = pd.concat([df.reset_index(drop=True), ratio_df], axis=1)
    del parts, ratio_df
    gc.collect()
    return df


def load_dataset(file_path, dataset_name, compute_ratios=False):
    """Load dataset from file and optionally compute ratios from SMILES."""
    if not os.path.exists(file_path):
        return None, f"File not found: {file_path}"

    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == ".csv":
            df = pd.read_csv(file_path)
        elif ext == ".tsv":
            df = pd.read_csv(file_path, sep="\t")
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)
        elif ext == ".sdf":
            from rdkit.Chem import PandasTools

            df = PandasTools.LoadSDF(file_path, smilesName="SMILES")
        else:
            return None, f"Unsupported file format: {ext}"
    except Exception as e:
        return None, f"Error loading file: {str(e)}"

    if "SMILES" not in df.columns:
        for col in df.columns:
            if "smiles" in col.lower() or "canonical" in col.lower():
                df.rename(columns={col: "SMILES"}, inplace=True)
                break

    if "SMILES" not in df.columns:
        return None, "No SMILES column found in dataset"

    df = df.dropna(subset=["SMILES"])
    df["SMILES"] = df["SMILES"].astype(str).str.strip()

    ratio_cols = [f"ratio_{e}" for e in DEFAULT_ELEMENTS]
    existing_ratios = [c for c in ratio_cols if c in df.columns]

    if not existing_ratios and compute_ratios and HAS_RDKIT:
        df = _compute_ratios_batched(df, dataset_name, DEFAULT_ELEMENTS)
    elif not existing_ratios and compute_ratios and not HAS_RDKIT:
        print("  WARNING: RDKit not available, cannot compute ratios")

    drop_cols = [
        c for c in df.columns if c.startswith("ratio_") and c not in ratio_cols
    ]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    keep_cols = (
        ["SMILES"]
        + [c for c in df.columns if c != "SMILES" and not c.startswith("ratio_")]
        + ratio_cols
    )
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    print(f"  Loaded {len(df)} molecules from {dataset_name}")
    return df, None


def remove_smiles_duplicates(df, dataset_name):
    """Remove duplicate SMILES from dataset."""
    before = len(df)
    df = df.drop_duplicates(subset=["SMILES"], keep="first")
    after = len(df)
    print(
        f"  {dataset_name}: {before} → {after} molecules ({before - after} duplicates removed)"
    )
    return df


# ============================================================================
# Twin Finding (8-ratio mode with block-based optimization)
# ============================================================================


def _gamma_fast(XB, rng, max_samples=MAX_GAMMA_SAMPLES):
    """Fast gamma estimation using sampling."""
    if len(XB) > max_samples:
        idx = rng.choice(len(XB), max_samples, replace=False)
        XB = XB[idx]
    sq = euclidean_distances(XB, XB, squared=True)
    nz = sq[sq > 0]
    med = np.median(nz) if nz.size > 0 else 1.0
    del sq, nz
    return 1.0 / max(med, 1e-6)


def _gamma_bulk(XB):
    """Bulk gamma estimation using median of squared distances across chunks."""
    if len(XB) > 10000:
        meds = []
        for i in range(0, len(XB), 2000):
            ch = XB[i : min(i + 2000, len(XB))]
            sq = euclidean_distances(ch, ch, squared=True)
            nz = sq[sq > 0]
            if nz.size:
                meds.append(np.median(nz))
            del sq, nz, ch
        med = np.median(meds) if meds else 1.0
    else:
        sq = euclidean_distances(XB, XB, squared=True)
        nz = sq[sq > 0]
        med = np.median(nz) if nz.size > 0 else 1.0
        del sq, nz
    return 1.0 / max(med, 1e-6)


def logit_transform(s):
    """Logit transform for similarity scores."""
    s_adj = np.clip(s - EPSILON, 1e-10, 1 - 1e-10)
    return np.log(s_adj / (1 - s_adj))


def prepare_df_for_twins(df, features):
    """Prepare dataframe with fp_block for block-based twin detection."""
    df = df.copy()
    df = df.reset_index(drop=True)
    valid_ratios = [c for c in features if c in df.columns]
    df["fp_block"] = df[valid_ratios].gt(0).astype(int).astype(str).agg("".join, axis=1)
    df[valid_ratios] = df[valid_ratios].fillna(0).astype(np.float32)
    return df


def find_all_twins(
    df_query, df_target, features, mode="fast", z_threshold=Z_THRESHOLD, seed_offset=0
):
    """Find all twin pairs between query and target datasets using block-based approach."""
    rng = _rng(GLOBAL_SEED + seed_offset)
    print(f"\n  Finding twins with {len(features)} features in {mode.upper()} mode...")

    ratio_cols = [c for c in features if c.startswith("ratio_")]
    if not ratio_cols:
        print("  ERROR: No ratio columns found!")
        return pd.DataFrame()

    df_query = prepare_df_for_twins(df_query, ratio_cols)
    df_target = prepare_df_for_twins(df_target, ratio_cols)

    gamma_fn = _gamma_fast if mode == "fast" else _gamma_bulk
    unique_fps = sorted(set(df_query["fp_block"]) & set(df_target["fp_block"]))

    print(
        f"  Mode: {mode.upper()} | {len(unique_fps)} shared blocks | {len(df_query)} queries vs {len(df_target)} targets"
    )

    all_pairs = []
    for i, fp in enumerate(unique_fps):
        ga = df_query[df_query["fp_block"] == fp]
        gb = df_target[df_target["fp_block"] == fp]
        if len(gb) < 5:
            continue

        XA = ga[ratio_cols].values.astype(np.float32)
        XB = gb[ratio_cols].values.astype(np.float32)

        if mode == "fast":
            gamma = _gamma_fast(XB, rng)
        else:
            gamma = _gamma_bulk(XB)

        for start in range(0, len(XA), CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, len(XA))
            XA_c = XA[start:end]
            sim = rbf_kernel(XA_c, XB, gamma=gamma).astype(np.float32)
            lt = logit_transform(sim)
            mu = lt.mean(axis=1, keepdims=True)
            sd = lt.std(axis=1, keepdims=True)
            z = (lt - mu) / (sd + 1e-9)
            rows, cols = np.where(z > z_threshold)
            for r, c in zip(rows, cols):
                all_pairs.append(
                    {
                        "query_idx": ga.index[start + r],
                        "target_idx": gb.index[c],
                        "similarity": float(sim[r, c]),
                        "z_score": float(z[r, c]),
                        "query_smiles": ga.iloc[start + r]["SMILES"],
                        "target_smiles": gb.iloc[c]["SMILES"],
                    }
                )
            del sim, lt, z
        del XA, XB
        if i % 10 == 0 or i == len(unique_fps) - 1:
            print(f"    Block {i + 1}/{len(unique_fps)} done")

    df_twins = pd.DataFrame(all_pairs)
    print(f"  Found {len(df_twins)} twin pairs")
    return df_twins


# ============================================================================
# Drug-likeness Filter
# ============================================================================


def compute_druglike_descriptors(smi):
    """Compute molecular descriptors for drug-likeness filter."""
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
            "NumRings": Descriptors.RingCount(mol),
            "FractionCSP3": Descriptors.FractionCSP3(mol),
            "NumHeavyAtoms": Descriptors.HeavyAtomCount(mol),
        }
    except Exception as e:
        return None


def apply_druglike_filter(df):
    """Apply Lipinski's rule of 5 filter."""
    if df.empty:
        return df

    need_computation = "MolWt" not in df.columns
    if not need_computation and "MolWt" in df.columns:
        need_computation = df["MolWt"].isna().all()

    if need_computation:
        print(f"  Computing molecular descriptors for {len(df)} molecules...")
        valid_indices = []
        desc_list = []
        for idx, smi in enumerate(df["SMILES"]):
            d = compute_druglike_descriptors(smi)
            if d is not None:
                valid_indices.append(idx)
                desc_list.append(d)

        print(
            f"  Successfully computed descriptors for {len(desc_list)}/{len(df)} molecules"
        )

        if not desc_list:
            print("  No valid molecules for drug-likeness filter")
            return df.iloc[:0].copy()

        df = df.iloc[valid_indices].reset_index(drop=True)
        desc_df = pd.DataFrame(desc_list, index=df.index)
        for col in desc_df.columns:
            df[col] = desc_df[col]

    rules = (
        (df["MolWt"] <= 1000)
        & (df["MolWt"] >= 100)
        & (df["LogP"] >= -2)
        & (df["LogP"] <= 10)
        & (df["NumHBD"] <= 6)
        & (df["NumHBA"] <= 15)
        & (df["TPSA"] <= 250)
        & (df["NumRotatableBonds"] <= 20)
    )

    df_druglike = df[rules].copy()
    print(f"  Drug-like molecules: {len(df)} → {len(df_druglike)}")
    return df_druglike


# ============================================================================
# Full Pipeline with pIC50 Prediction
# ============================================================================


def run_twin_pipeline(
    query_path,
    target_path,
    output_dir=None,
    z_threshold=Z_THRESHOLD,
    model_path=None,
    predict_high_activity=True,
    high_activity_threshold=6.5,
    mode="fast",
):
    """Run the complete twin detection pipeline with optional pIC50 prediction."""

    if output_dir is None:
        output_dir = os.path.join(
            BASE_DIR, "results", datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"TwinSAR-8 Pipeline - {mode.upper()} Mode (8 Single-Element Ratios)")
    print(f"{'=' * 60}\n")

    print(f"Query:   {query_path}")
    print(f"Target:  {target_path}")
    print(f"Output:  {output_dir}")
    print(f"Model:   {model_path if model_path else 'None'}")
    print(f"Mode:    {mode.upper()}")

    predictor = None
    if model_path:
        predictor = get_model_for_prediction(model_path)
        if predictor is None:
            print("  WARNING: Could not load model, skipping pIC50 prediction")
        else:
            print(f"  Model loaded: {model_path}")

    print("\n[1/7] Loading datasets...")
    df_query, err = load_dataset(query_path, "query", compute_ratios=False)
    if err:
        print(f"ERROR: {err}")
        return None

    df_target, err = load_dataset(target_path, "target", compute_ratios=False)
    if err:
        print(f"ERROR: {err}")
        return None

    print("\n[2/7] Removing duplicates...")
    df_query = remove_smiles_duplicates(df_query, "query")
    df_target = remove_smiles_duplicates(df_target, "target")

    pic50_col = None
    for col in df_query.columns:
        if "pic50" in col.lower():
            pic50_col = col
            break
    if pic50_col is not None:
        before_filter = len(df_query)
        df_query = df_query[df_query[pic50_col] > high_activity_threshold].reset_index(
            drop=True
        )
        print(
            f"\n  High-potency query filter (pIC50 > {high_activity_threshold}): "
            f"{before_filter} -> {len(df_query)} molecules"
        )
        if len(df_query) == 0:
            print("  ERROR: No query molecules pass the high-potency filter!")
            return None
    else:
        print(
            f"\n  WARNING: No pIC50 column found in query file -- "
            f"skipping high-potency filter. Add a 'pIC50' column to filter queries."
        )

    if "ID" not in df_query.columns:
        df_query["ID"] = [f"Q_{i}" for i in range(len(df_query))]
    if "ID" not in df_target.columns:
        df_target["ID"] = [f"T_{i}" for i in range(len(df_target))]

    features = [f"ratio_{e}" for e in DEFAULT_ELEMENTS]
    missing = [
        f for f in features if f not in df_query.columns or f not in df_target.columns
    ]
    if missing:
        print(f"  ERROR: Missing ratio columns after ratio computation: {missing}")
        return None
    print(
        f"\n[3/7] Finding twins using {len(features)} ratio features: {DEFAULT_ELEMENTS}..."
    )
    seed_offset = 0 if mode == "fast" else 500
    df_twins = find_all_twins(
        df_query,
        df_target,
        features,
        mode=mode,
        z_threshold=z_threshold,
        seed_offset=seed_offset,
    )

    if len(df_twins) == 0:
        print("\n  No twins found!")
        return None

    print("\n[4/7] Extracting unique twin targets...")
    unique_targets = df_twins[["target_idx", "target_smiles"]].drop_duplicates()
    df_unique = df_target.iloc[unique_targets["target_idx"]].reset_index(drop=True)
    print(f"  Unique twin targets: {len(df_unique)}")

    print("\n[5/7] Applying drug-likeness filter...")
    df_druglike = apply_druglike_filter(df_unique)

    df_high_activity = None
    if predictor and predict_high_activity:
        print(f"\n[6/7] Predicting pIC50 for {len(df_druglike)} drug-like molecules...")
        smiles_list = df_druglike["SMILES"].tolist()
        predictions = predictor.predict(smiles_list)

        if predictions is not None:
            df_druglike = df_druglike.copy()
            df_druglike["predicted_pIC50"] = predictions

            high_activity_mask = predictions > high_activity_threshold
            df_high_activity = df_druglike[high_activity_mask].copy()
            print(
                f"  High-activity hits (pIC50 > {high_activity_threshold}): {len(df_high_activity)}"
            )
        else:
            print("  WARNING: pIC50 prediction failed")
            df_druglike["predicted_pIC50"] = np.nan
    else:
        df_druglike["predicted_pIC50"] = np.nan

    print(f"\n[7/7] Saving results...")
    twins_csv = os.path.join(output_dir, "all_twin_pairs.csv")
    df_twins.to_csv(twins_csv, index=False)
    print(f"  Saved: {twins_csv}")

    unique_csv = os.path.join(output_dir, "unique_twin_targets.csv")
    df_unique.to_csv(unique_csv, index=False)
    print(f"  Saved: {unique_csv}")

    druglike_csv = os.path.join(output_dir, "druglike_twins.csv")
    df_druglike.to_csv(druglike_csv, index=False)
    print(f"  Saved: {druglike_csv}")

    if df_high_activity is not None and len(df_high_activity) > 0:
        high_csv = os.path.join(output_dir, "high_activity_twins.csv")
        df_high_activity.to_csv(high_csv, index=False)
        print(f"  Saved: {high_csv}")

    stats = {
        "total_twin_pairs": len(df_twins),
        "unique_targets": len(df_unique),
        "druglike_molecules": len(df_druglike),
        "high_activity_hits": len(df_high_activity)
        if df_high_activity is not None
        else 0,
        "z_threshold": z_threshold,
    }

    stats_file = os.path.join(output_dir, "pipeline_stats.json")
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved: {stats_file}")

    print(f"\n{'=' * 60}")
    print("Pipeline completed!")
    print(f"{'=' * 60}")
    print(f"  Twin pairs found:      {stats['total_twin_pairs']}")
    print(f"  Unique twin targets:  {stats['unique_targets']}")
    print(f"  Drug-like molecules:  {stats['druglike_molecules']}")
    if predictor:
        print(f"  High-activity hits:   {stats['high_activity_hits']}")
    print(f"{'=' * 60}\n")

    return {
        "twins": df_twins,
        "unique_targets": df_unique,
        "druglike": df_druglike,
        "high_activity": df_high_activity,
        "stats": stats,
        "output_dir": output_dir,
    }


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="TwinSAR-8: Full pipeline with ML prediction"
    )
    parser.add_argument("--query", required=True, help="Path to query CSV file")
    parser.add_argument("--target", required=True, help="Path to target CSV file")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=Z_THRESHOLD,
        help="Z-score threshold (default: 2.576)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="fast",
        choices=["fast", "bulk"],
        help="Twin detection mode: fast or bulk (default: fast)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to pIC50 prediction model (.joblib file)",
    )
    parser.add_argument(
        "--no-predict",
        action="store_true",
        help="Skip pIC50 prediction even if model is available",
    )
    parser.add_argument(
        "--high-activity-threshold",
        type=float,
        default=6.5,
        help="Threshold for high-activity flag (default: 6.5)",
    )
    parser.add_argument("--web", action="store_true", help="Run web interface")
    args = parser.parse_args()

    if args.web:
        run_web_interface()
    else:
        result = run_twin_pipeline(
            args.query,
            args.target,
            args.output,
            z_threshold=args.z_threshold,
            model_path=args.model,
            predict_high_activity=not args.no_predict,
            high_activity_threshold=args.high_activity_threshold,
            mode=args.mode,
        )
        if result:
            print(f"\n✓ Pipeline completed successfully!")


# ============================================================================
# Web Interface
# ============================================================================


def run_web_interface():
    try:
        from flask import Flask, render_template, request, send_file
    except ImportError:
        print("ERROR: Flask not installed. Run: pip install flask")
        return

    app = Flask(__name__)
    app.config["UPLOAD_FOLDER"] = UPLOAD_DIR

    @app.route("/")
    def index():
        return """
        <html><head><title>TwinSAR-8</title>
        <style>
            body { font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; }
            h1 { color: #333; }
            .form-group { margin: 15px 0; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            input[type="file"], input[type="text"] { padding: 10px; border: 1px solid #ccc; width: 100%; }
            button { background: #007bff; color: white; padding: 12px 24px; border: none; cursor: pointer; }
            button:hover { background: #0056b3; }
            .help { font-size: 0.9em; color: #666; }
        </style>
        </head>
        <body>
            <h1>TwinSAR-8 - Full ML Pipeline</h1>
            <p>Molecular twin detection with pIC50 prediction.</p>
            <form action="/run" method="post" enctype="multipart/form-data">
                <div class="form-group">
                    <label>Query File (CSV with SMILES):</label>
                    <input type="file" name="query" required>
                </div>
                <div class="form-group">
                    <label>Target File (CSV with SMILES):</label>
                    <input type="file" name="target" required>
                </div>
                <div class="form-group">
                    <label>Model File (optional, .joblib):</label>
                    <input type="file" name="model">
                    <div class="help">Leave empty to skip pIC50 prediction</div>
                </div>
                <button type="submit">Run Pipeline</button>
            </form>
        </body></html>
        """

    @app.route("/run", methods=["POST"])
    def run():
        query_file = request.files["query"]
        target_file = request.files["target"]
        model_file = request.files.get("model")

        query_path = os.path.join(UPLOAD_DIR, query_file.filename)
        target_path = os.path.join(UPLOAD_DIR, target_file.filename)

        query_file.save(query_path)
        target_file.save(target_path)

        model_path = None
        if model_file and model_file.filename:
            model_path = os.path.join(UPLOAD_DIR, model_file.filename)
            model_file.save(model_path)

        result = run_twin_pipeline(query_path, target_path, model_path=model_path)

        if result:
            stats = result["stats"]
            return f"""
            <html><body>
            <h2>Results</h2>
            <ul>
                <li>Twin pairs: {stats["total_twin_pairs"]}</li>
                <li>Unique targets: {stats["unique_targets"]}</li>
                <li>Drug-like: {stats["druglike_molecules"]}</li>
                <li>High-activity: {stats["high_activity_hits"]}</li>
            </ul>
            <p>Results saved to: {result["output_dir"]}</p>
            </body></html>
            """
        return "No twins found"

    print("Starting TwinSAR-8 web interface on http://localhost:5000")
    app.run(debug=True, port=5000)


if __name__ == "__main__":
    main()
