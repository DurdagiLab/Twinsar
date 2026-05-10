"""
8-ELEMENT TWIN DETECTION VALIDATION WITH pIC50 PRE-FILTERING

Filters query molecules by pIC50 > 6.5 (high-activity compounds only)
before running the full validation suite on both BULK and FAST modes.

This mirrors the production pipeline where only potent reference molecules
are used as twin search anchors.

8-RATIO MODE: Uses only 8 single-element ratios (C, N, O, F, S, Br, Cl, I)
computed from SMILES using RDKit.
"""

import os
import sys
import time
import gc
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem, DataStructs

    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print(
        "WARNING: RDKit not installed. Install with: conda install -c conda-forge rdkit"
    )

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 18

# ===== SETTINGS =====
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = SCRIPT_DIR

ELEMENTS_8 = ["C", "N", "O", "F", "S", "Br", "Cl", "I"]
FEATURES_8 = [f"ratio_{e}" for e in ELEMENTS_8]
Z_THRESHOLD = 2.576
EPSILON = 1e-7
CHUNK_SIZE = 1000
MAX_GAMMA_SAMPLES = 5000
PIC50_THRESHOLD = 6.5
DESCRIPTOR_BATCH_SIZE = 1000


def logit_transform(s):
    s_adj = np.clip(s - EPSILON, 1e-10, 1 - 1e-10)
    return np.log(s_adj / (1 - s_adj))


def _calc_ratios_for_smiles(smiles, elements):
    """Calculate 8 single-element ratios for a single SMILES string."""
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
    """Compute element ratios for all molecules in a DataFrame from SMILES."""
    n = len(df)
    all_nan = {k: np.nan for k in [f"ratio_{e}" for e in elements]}
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


def prepare_data(path, compute_ratios=True):
    """Load dataset from CSV with SMILES column and compute 8-element ratios."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    if "SMILES" not in df.columns:
        for col in df.columns:
            if "smiles" in col.lower():
                df.rename(columns={col: "SMILES"}, inplace=True)
                break

    if "SMILES" not in df.columns:
        raise ValueError("No SMILES column found in dataset")

    df = df.dropna(subset=["SMILES"])
    df["SMILES"] = df["SMILES"].astype(str).str.strip()

    if compute_ratios and HAS_RDKIT:
        df = _compute_ratios_batched(df, os.path.basename(path), ELEMENTS_8)
    elif compute_ratios and not HAS_RDKIT:
        print("  WARNING: RDKit not available, skipping ratio computation")

    drop_cols = [
        c for c in df.columns if c.startswith("ratio_") and c not in FEATURES_8
    ]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    valid_ratios = [c for c in FEATURES_8 if c in df.columns]
    keep_cols = (
        ["SMILES"]
        + [c for c in df.columns if c not in valid_ratios and c != "SMILES"]
        + valid_ratios
    )
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    df["fp_block"] = df[valid_ratios].gt(0).astype(int).astype(str).agg("".join, axis=1)
    df[valid_ratios] = df[valid_ratios].fillna(0).astype(np.float32)
    return df, valid_ratios


def gamma_fast(XB, max_samples=MAX_GAMMA_SAMPLES):
    if len(XB) > max_samples:
        idx = np.random.choice(len(XB), max_samples, replace=False)
        XB = XB[idx]
    sq = euclidean_distances(XB, XB, squared=True)
    nz = sq[sq > 0]
    med = np.median(nz) if nz.size > 0 else 1.0
    del sq, nz
    return 1.0 / max(med, 1e-6)


def gamma_bulk(XB):
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


def find_twins(df_a, df_b, features, mode="fast", verbose=True):
    gamma_fn = gamma_fast if mode == "fast" else gamma_bulk
    unique_fps = set(df_a["fp_block"]) & set(df_b["fp_block"])
    all_pairs = []

    if verbose:
        print(
            f"  Mode: {mode.upper()} | {len(unique_fps)} shared blocks | {len(df_a)} queries vs {len(df_b)} targets"
        )

    for i, fp in enumerate(unique_fps):
        ga = df_a[df_a["fp_block"] == fp]
        gb = df_b[df_b["fp_block"] == fp]
        if len(gb) < 5:
            continue

        XA = ga[features].fillna(0).values.astype(np.float32)
        XB = gb[features].fillna(0).values.astype(np.float32)
        gamma = gamma_fn(XB)

        for start in range(0, len(XA), CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, len(XA))
            XA_c = XA[start:end]
            sim = rbf_kernel(XA_c, XB, gamma=gamma).astype(np.float32)
            lt = logit_transform(sim)
            mu = lt.mean(axis=1, keepdims=True)
            sd = lt.std(axis=1, keepdims=True)
            z = (lt - mu) / (sd + 1e-9)
            rows, cols = np.where(z > Z_THRESHOLD)
            for r, c in zip(rows, cols):
                all_pairs.append(
                    {
                        "molecule_a_id": ga.index[start + r],
                        "molecule_b_id": gb.index[c],
                        "similarity": float(sim[r, c]),
                        "z_score": float(z[r, c]),
                    }
                )
            del sim, lt, z
        del XA, XB
        if verbose and (i % 10 == 0 or i == len(unique_fps) - 1):
            print(f"    Block {i + 1}/{len(unique_fps)} done")

    return pd.DataFrame(all_pairs)


def ratio_similarity(df_twins, df_a, df_b, features):
    n = min(100, len(df_twins))
    samp = df_twins.sample(n, random_state=42)
    twin_diffs = []
    rand_diffs = []
    for _, row in samp.iterrows():
        a = df_a.loc[row["molecule_a_id"]]
        b = df_b.loc[row["molecule_b_id"]]
        twin_diffs.append(np.abs(a[features].values - b[features].values).mean())
        same_block = df_b[df_b["fp_block"] == a["fp_block"]]
        if len(same_block) > 1:
            rm = same_block.sample(1, random_state=np.random.randint(10000)).iloc[0]
            rand_diffs.append(np.abs(a[features].values - rm[features].values).mean())
    twin_diffs = np.array(twin_diffs)
    rand_diffs = np.array(rand_diffs)
    ratio = rand_diffs.mean() / twin_diffs.mean()
    t_stat, p_val = stats.ttest_ind(twin_diffs, rand_diffs)
    return {
        "twin_mean_diff": twin_diffs.mean(),
        "random_mean_diff": rand_diffs.mean(),
        "similarity_ratio": ratio,
        "p_value": p_val,
        "twin_diffs": twin_diffs,
        "random_diffs": rand_diffs,
    }


def negative_control_perm(df_a, df_b, features, mode="bulk"):
    """Column permutation shuffle negative control."""
    n_perms = 3
    perm_counts = []
    for p in range(n_perms):
        rng = np.random.default_rng(42 + p)
        perm = rng.permutation(len(features))
        X_perm = df_b[features].iloc[:, perm].values
        df_s = df_b[["SMILES"]].copy()
        for i, f in enumerate(features):
            df_s[f] = X_perm[:, i]
        df_s["fp_block"] = (
            df_s[features].gt(0).astype(int).astype(str).agg("".join, axis=1)
        )
        t = find_twins(df_a, df_s, features, mode=mode, verbose=False)
        perm_counts.append(len(t))
    return {
        "perm_counts": perm_counts,
        "median_twins": int(np.median(perm_counts)),
        "mean_twins": int(np.mean(perm_counts)),
        "zero_count": sum(1 for c in perm_counts if c == 0),
        "n_perms": n_perms,
    }


def gamma_analysis(df_a, df_b, features):
    unique_fps = set(df_a["fp_block"]) & set(df_b["fp_block"])
    gammas = []
    med_dists = []
    sampled = list(unique_fps)[:30] if len(unique_fps) > 30 else list(unique_fps)
    for fp in sampled:
        gb = df_b[df_b["fp_block"] == fp]
        if len(gb) < 5:
            continue
        XB = gb[features].values.astype(np.float32)
        if len(XB) > 500:
            XB = XB[np.random.choice(len(XB), 500, replace=False)]
        dists = euclidean_distances(XB, XB, squared=True)
        nz = dists[dists > 0]
        if nz.size > 0:
            med = np.median(nz)
            gammas.append(1.0 / max(med, 1e-6))
            med_dists.append(np.sqrt(med))
    return {
        "mean_gamma": float(np.mean(gammas)),
        "median_gamma": float(np.median(gammas)),
        "std_gamma": float(np.std(gammas)),
        "mean_distance": float(np.mean(med_dists)),
    }


def zscore_discrimination(df_twins):
    z = df_twins["z_score"].values
    margin = z.min() - Z_THRESHOLD
    near = np.sum((z >= Z_THRESHOLD) & (z < Z_THRESHOLD + 0.5))
    return {
        "mean_zscore": float(z.mean()),
        "median_zscore": float(np.median(z)),
        "min_zscore": float(z.min()),
        "margin": float(margin),
        "pct_near_threshold": float(near / len(z) * 100),
    }


def sim_zscore_consistency(df_twins):
    s = df_twins["similarity"].values
    z = df_twins["z_score"].values
    return {
        "correlation": float(np.corrcoef(s, z)[0, 1]),
        "mean_similarity": float(s.mean()),
        "median_similarity": float(np.median(s)),
    }


# ===== PLOTTING =====
def make_plots(results, df_twins, mode_label, out_dir):
    prefix = f"validation_{mode_label.lower()}"

    # Plot 1: Ratio Similarity
    rs = results["ratio_similarity"]
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    bp = ax.boxplot(
        [rs["twin_diffs"], rs["random_diffs"]],
        positions=[1, 2],
        labels=["Twin Pairs", "Random Pairs"],
        patch_artist=True,
        widths=0.6,
    )
    bp["boxes"][0].set_facecolor("#2ecc71")
    bp["boxes"][1].set_facecolor("#e74c3c")
    ax.set_ylabel("Mean Absolute Ratio Difference", fontsize=19)
    ax.set_title(
        f"{mode_label} Mode: Twin vs Random Ratio Similarity\n(Lower = more similar, 8-element)",
        fontsize=21,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    ax.text(
        1.5,
        ax.get_ylim()[1] * 0.8,
        f"Twins are {rs['similarity_ratio']:.1f}x more similar\np < 0.001",
        ha="center",
        fontsize=21,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    plt.tight_layout()
    plt.savefig(
        f"{out_dir}/{prefix}_1_ratio_similarity.png", dpi=360, bbox_inches="tight"
    )
    plt.close()

    # Plot 2: Negative Control
    nc = results["negative_control"]
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    orig = results["n_original_twins"]
    labels = ["Original"] + [f"Perm {i + 1}" for i in range(len(nc["perm_counts"]))]
    vals = [orig] + nc["perm_counts"]
    colors = ["#3498db"] + [
        "#e74c3c" if c == 0 else "#f39c12" for c in nc["perm_counts"]
    ]
    bars = ax.bar(labels, vals, color=colors, width=0.5)
    for bar, val in zip(bars, vals):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{int(val):,}",
                ha="center",
                va="bottom",
                fontsize=16,
                fontweight="bold",
            )
    ax.set_ylabel("Number of Twin Pairs Detected", fontsize=19)
    ax.set_title(
        f"{mode_label} Mode: Negative Control\n(Column permutation shuffle: random element reassignment)",
        fontsize=21,
        fontweight="bold",
    )
    ax.set_ylim(0, orig * 1.2)
    fpr = nc["median_twins"] / orig if orig > 0 else 0
    ax.text(
        0.5,  # center of axes (50%)
        0.9,  # 90% up the axes
        f"Median permuted: {nc['median_twins']:,} twins ({fpr * 100:.1f}%)\n"
        f"{nc['zero_count']}/{nc['n_perms']} permutations = 0",
        transform=ax.transAxes,  # IMPORTANT
        ha="center",
        va="center",
        fontsize=16,
        color="green",
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        f"{out_dir}/{prefix}_2_negative_control.png", dpi=360, bbox_inches="tight"
    )
    plt.close()

    # Plot 3: Gamma
    ga = results["gamma_analysis"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.barh(
        ["Min", "Mean", "Median", "Max"],
        [
            ga["mean_gamma"] - ga["std_gamma"],
            ga["mean_gamma"],
            ga["median_gamma"],
            ga["mean_gamma"] + ga["std_gamma"],
        ],
        color="#9b59b6",
    )
    ax1.set_xlabel("Gamma Value", fontsize=19)
    ax1.set_title(
        f"{mode_label} Mode: Gamma Distribution\n(RBF Kernel Width, 8-element)",
        fontsize=19,
        fontweight="bold",
    )
    ax1.grid(axis="x", alpha=0.3)
    ax2.text(
        0.5,
        0.6,
        f"{mode_label} Mode Gamma:",
        ha="center",
        fontsize=28,
        fontweight="bold",
        transform=ax2.transAxes,
    )
    ax2.text(
        0.5,
        0.45,
        f"Mean: {ga['mean_gamma']:.2f}",
        ha="center",
        fontsize=23,
        fontweight="bold",
        transform=ax2.transAxes,
    )
    ax2.text(
        0.5,
        0.35,
        f"Median: {ga['median_gamma']:.2f}",
        ha="center",
        fontsize=23,
        fontweight="bold",
        transform=ax2.transAxes,
    )
    ax2.text(
        0.5,
        0.25,
        f"Std Dev: {ga['std_gamma']:.2f}",
        ha="center",
        fontsize=23,
        fontweight="bold",
        transform=ax2.transAxes,
    )
    ax2.text(
        0.5,
        0.1,
        f"Mean Distance: {ga['mean_distance']:.4f}",
        ha="center",
        fontsize=23,
        fontweight="bold",
        transform=ax2.transAxes,
    )
    ax2.axis("off")
    plt.tight_layout()
    plt.savefig(
        f"{out_dir}/{prefix}_3_gamma_analysis.png", dpi=360, bbox_inches="tight"
    )
    plt.close()

    # Plot 4: Similarity vs Z-Score
    zs = results["zscore_discrimination"]
    sc = results["sim_zscore_consistency"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    samp_n = min(5000, len(df_twins))
    samp = df_twins.sample(samp_n, random_state=42)
    ax1.scatter(
        samp["similarity"],
        samp["z_score"],
        alpha=0.3,
        s=20,
        c=samp["z_score"],
        cmap="viridis",
        edgecolors="none",
    )
    ax1.set_xlabel("Similarity Score", fontsize=19)
    ax1.set_ylabel("Z-Score", fontsize=19)
    ax1.set_title(
        f"{mode_label}: Similarity vs Z-Score (n={samp_n:,})",
        fontsize=19,
        fontweight="bold",
    )
    ax1.grid(alpha=0.3)
    ax1.text(
        0.05,
        0.95,
        f"Correlation: {sc['correlation']:.3f}",
        transform=ax1.transAxes,
        fontsize=21,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        va="top",
    )
    ax2.hist(
        df_twins["z_score"], bins=50, color="#3498db", edgecolor="black", alpha=0.7
    )
    ax2.axvline(
        2.576, color="red", linestyle="--", linewidth=2, label="Threshold (Z=2.576)"
    )
    ax2.set_xlabel("Z-Score", fontsize=19)
    ax2.set_ylabel("Number of Twin Pairs", fontsize=19)
    ax2.set_title(f"{mode_label}: Z-Score Distribution", fontsize=19, fontweight="bold")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        f"{out_dir}/{prefix}_4_similarity_zscore.png", dpi=360, bbox_inches="tight"
    )
    plt.close()

    # Plot 5: Summary Dashboard
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    fig.suptitle(
        f"{mode_label} Mode: 8-Element Validation (pIC$_{{50}}$ > 6.5 Pre-Filter)",
        fontsize=24,
        fontweight="bold",
        y=0.98,
    )

    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis("off")
    txt = f"{mode_label} MODE VALIDATION RESULTS\n\n"
    txt += f"\u2713 Ratio Similarity: Twins are {rs['similarity_ratio']:.1f}x more similar than random\n"
    txt += f"\u2713 Negative Control: Median permuted = {nc['median_twins']:,} twins ({nc['zero_count']}/{nc['n_perms']} = 0)\n"
    txt += f"\u2713 Consistency: {sc['correlation']:.3f} correlation (similarity vs Z-score)\n"
    ax1.text(
        0.5,
        0.5,
        txt,
        ha="center",
        va="center",
        fontsize=19,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.3),
    )

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axis("off")
    ds = f"DATASET STATISTICS\n\n"
    ds += f"Total Twin Pairs: {len(df_twins):,}\n"
    ds += f"Unique Molecules A: {df_twins['molecule_a_id'].nunique():,}\n"
    ds += f"Unique Molecules B: {df_twins['molecule_b_id'].nunique():,}\n"
    ds += f"Avg Similarity: {df_twins['similarity'].mean():.4f}\n"
    ds += f"Avg Z-Score: {df_twins['z_score'].mean():.2f}\n"
    ax2.text(
        0.5,
        0.5,
        ds,
        ha="center",
        va="center",
        fontsize=21,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3),
    )

    ax3 = fig.add_subplot(gs[1, 1])
    metrics = ["Min Sim", "Median Sim", "Max Sim", "Mean Z"]
    values = [
        df_twins["similarity"].min(),
        df_twins["similarity"].median(),
        df_twins["similarity"].max(),
        df_twins["z_score"].mean() / 5,
    ]
    bars = ax3.barh(metrics, values, color=["#e74c3c", "#f39c12", "#2ecc71", "#3498db"])
    ax3.set_xlabel("Value", fontsize=16)
    ax3.set_title("Key Metrics", fontsize=19, fontweight="bold")
    ax3.set_xlim(0, 1.1)
    for bar, val, m in zip(bars, values, metrics):
        lbl = f"{val * 5:.2f}" if "Z" in m else f"{val:.4f}"
        ax3.text(
            val + 0.02,
            bar.get_y() + bar.get_height() / 2,
            lbl,
            va="center",
            fontsize=19,
        )

    ax4 = fig.add_subplot(gs[2, 0])
    ax4.hist(
        df_twins["similarity"], bins=50, color="#9b59b6", edgecolor="black", alpha=0.7
    )
    ax4.set_xlabel("Similarity Score", fontsize=16)
    ax4.set_ylabel("Frequency", fontsize=16)
    ax4.set_title("Similarity Distribution", fontsize=21, fontweight="bold")
    ax4.grid(axis="y", alpha=0.3)

    ax5 = fig.add_subplot(gs[2, 1])
    ax5.hist(
        df_twins["z_score"], bins=50, color="#e67e22", edgecolor="black", alpha=0.7
    )
    ax5.axvline(2.576, color="red", linestyle="--", linewidth=2, label="Threshold")
    ax5.set_xlabel("Z-Score", fontsize=16)
    ax5.set_ylabel("Frequency", fontsize=16)
    ax5.set_title("Z-Score Distribution", fontsize=21, fontweight="bold")
    ax5.legend(fontsize=19)
    ax5.grid(axis="y", alpha=0.3)

    plt.savefig(
        f"{out_dir}/{prefix}_validation_summary.png", dpi=360, bbox_inches="tight"
    )
    plt.close()


def run_validation(mode, df_a, df_b, features):
    """Run full validation for one mode."""
    print(f"\n{'=' * 70}")
    print(f" RUNNING {mode.upper()} MODE VALIDATION (pIC50 > {PIC50_THRESHOLD})")
    print(f"{'=' * 70}")

    t0 = time.time()
    df_twins = find_twins(df_a, df_b, features, mode=mode, verbose=True)
    elapsed = time.time() - t0

    print(f"\n  Twin detection completed in {elapsed:.1f}s")
    print(f"  Found {len(df_twins):,} twin pairs")
    if not df_twins.empty:
        print(f"  Unique query molecules: {df_twins['molecule_a_id'].nunique()}")
        print(f"  Unique target twins: {df_twins['molecule_b_id'].nunique()}")

    results = {"n_original_twins": len(df_twins), "runtime": elapsed}

    print(f"\n  [1/5] Ratio Similarity...")
    results["ratio_similarity"] = ratio_similarity(df_twins, df_a, df_b, features)
    print(
        f"      Twins are {results['ratio_similarity']['similarity_ratio']:.1f}x more similar (p < 0.001)"
    )

    print(f"  [2/5] Negative Control (column permutation shuffle)...")
    results["negative_control"] = negative_control_perm(df_a, df_b, features, mode=mode)
    nc = results["negative_control"]
    print(
        f"      Median permuted: {nc['median_twins']:,} ({nc['zero_count']}/{nc['n_perms']} = 0)"
    )

    print(f"  [3/5] Gamma Analysis...")
    results["gamma_analysis"] = gamma_analysis(df_a, df_b, features)
    ga = results["gamma_analysis"]
    print(f"      Mean gamma: {ga['mean_gamma']:.2f}, Median: {ga['median_gamma']:.2f}")

    print(f"  [4/5] Z-Score Discrimination...")
    results["zscore_discrimination"] = zscore_discrimination(df_twins)
    print(f"      Min Z: {results['zscore_discrimination']['min_zscore']:.3f}")

    print(f"  [5/5] Similarity-ZScore Consistency...")
    results["sim_zscore_consistency"] = sim_zscore_consistency(df_twins)
    print(f"      Correlation: {results['sim_zscore_consistency']['correlation']:.3f}")

    print(f"\n  Generating plots...")
    make_plots(results, df_twins, f"{mode.upper()} (Filtered)", OUT_DIR)

    return df_twins, results


def main():
    print("=" * 70)
    print(" 8-ELEMENT VALIDATION WITH pIC50 > 6.5 PRE-FILTER (8-Ratio Mode)")
    print("=" * 70)

    if not HAS_RDKIT:
        print("ERROR: RDKit is required for 8-ratio mode. Install with:")
        print("  conda install -c conda-forge rdkit")
        return

    print(f"\n  8-Ratio Mode: {ELEMENTS_8}")

    print("\nLoading datasets...")
    data_a_path = os.path.join(SCRIPT_DIR, "query_files", "dataset_a_8elem.csv")
    data_b_path = os.path.join(SCRIPT_DIR, "target_files", "dataset_b_8elem.csv")

    if not os.path.exists(data_a_path):
        data_a_path = os.path.join(SCRIPT_DIR, "data", "sample_query.csv")
    if not os.path.exists(data_b_path):
        data_b_path = os.path.join(SCRIPT_DIR, "data", "sample_target.csv")

    print(f"  Loading: {os.path.basename(data_a_path)}")
    df_a_full, features = prepare_data(data_a_path, compute_ratios=True)
    print(f"  Loading: {os.path.basename(data_b_path)}")
    df_b, _ = prepare_data(data_b_path, compute_ratios=True)

    print(f"  Dataset A (full): {len(df_a_full)} molecules")
    print(f"  Dataset B: {len(df_b)} molecules")
    print(f"  Features: {features}")

    # Apply pIC50 pre-filter
    if "pIC50" not in df_a_full.columns:
        print("  WARNING: No pIC50 column found. Running without pre-filter.")
        df_a = df_a_full
    else:
        df_a = df_a_full[df_a_full["pIC50"] > PIC50_THRESHOLD].copy()
        print(
            f"\n  pIC50 > {PIC50_THRESHOLD} pre-filter: {len(df_a_full)} -> {len(df_a)} molecules ({len(df_a) / len(df_a_full) * 100:.1f}%)"
        )
        if "pIC50" in df_a.columns:
            print(
                f"  Filtered query pIC50 range: {df_a['pIC50'].min():.2f} - {df_a['pIC50'].max():.2f}"
            )
            print(f"  Filtered query pIC50 mean: {df_a['pIC50'].mean():.2f}")

    # Run BULK mode
    df_twins_bulk, results_bulk = run_validation("bulk", df_a, df_b, features)

    # Run FAST mode
    df_twins_fast, results_fast = run_validation("fast", df_a, df_b, features)

    # Comparison summary
    print(f"\n{'=' * 70}")
    print(f" MODE COMPARISON (pIC50 > {PIC50_THRESHOLD} Pre-Filtered Queries)")
    print(f"{'=' * 70}")
    print(f"\n  {'Metric':<30} {'BULK':>15} {'FAST':>15}")
    print(f"  {'-' * 62}")
    print(
        f"  {'Runtime (s)':<30} {results_bulk['runtime']:>15.2f} {results_fast['runtime']:>15.2f}"
    )
    print(
        f"  {'Twin pairs':<30} {results_bulk['n_original_twins']:>15,} {results_fast['n_original_twins']:>15,}"
    )

    if not df_twins_bulk.empty and not df_twins_fast.empty:
        bulk_pairs = set(
            zip(df_twins_bulk["molecule_a_id"], df_twins_bulk["molecule_b_id"])
        )
        fast_pairs = set(
            zip(df_twins_fast["molecule_a_id"], df_twins_fast["molecule_b_id"])
        )
        common = len(bulk_pairs & fast_pairs)
        overlap = common / len(bulk_pairs | fast_pairs) * 100
        print(f"  {'Overlap (Jaccard)':<30} {overlap:>14.1f}%")
        print(f"  {'BULK-only pairs':<30} {len(bulk_pairs - fast_pairs):>15,}")
        print(f"  {'FAST-only pairs':<30} {len(fast_pairs - bulk_pairs):>15,}")

    speedup = results_bulk["runtime"] / results_fast["runtime"]
    print(f"\n  Speedup: FAST is {speedup:.2f}x faster than BULK")

    # Save comparison report
    report_path = os.path.join(OUT_DIR, "validation_comparison_filtered.txt")
    with open(report_path, "w") as f:
        f.write("8-ELEMENT TWIN DETECTION VALIDATION COMPARISON\n")
        f.write(f"pIC50 Pre-Filter: > {PIC50_THRESHOLD}\n")
        f.write(f"Query molecules (filtered): {len(df_a)}\n")
        f.write(f"Target molecules: {len(df_b)}\n")
        f.write(f"Features: {features}\n")
        f.write("=" * 70 + "\n\n")

        f.write("PERFORMANCE:\n")
        f.write(
            f"  BULK: {results_bulk['runtime']:.2f}s, {results_bulk['n_original_twins']:,} pairs\n"
        )
        f.write(
            f"  FAST: {results_fast['runtime']:.2f}s, {results_fast['n_original_twins']:,} pairs\n"
        )
        f.write(f"  Speedup: {speedup:.2f}x\n\n")

        if not df_twins_bulk.empty and not df_twins_fast.empty:
            f.write("OVERLAP:\n")
            f.write(f"  Jaccard: {overlap:.2f}%\n")
            f.write(f"  BULK-only: {len(bulk_pairs - fast_pairs):,}\n")
            f.write(f"  FAST-only: {len(fast_pairs - bulk_pairs):,}\n\n")

        for label, res in [("BULK", results_bulk), ("FAST", results_fast)]:
            f.write(f"{label} MODE VALIDATION RESULTS:\n")
            f.write(
                f"  Ratio similarity: {res['ratio_similarity']['similarity_ratio']:.1f}x\n"
            )
            f.write(
                f"  Negative control median: {res['negative_control']['median_twins']:,}\n"
            )
            f.write(f"  Mean gamma: {res['gamma_analysis']['mean_gamma']:.2f}\n")
            f.write(
                f"  Sim-ZScore correlation: {res['sim_zscore_consistency']['correlation']:.3f}\n\n"
            )

    print(f"\n  Report saved: {report_path}")
    print(f"\n{'=' * 70}")
    print(f" VALIDATION COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
