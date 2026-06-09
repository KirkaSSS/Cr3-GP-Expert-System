"""
Gaussian Process Model for Cr³⁺ Phosphor Dq/B Prediction
Author: Snežana Đurković
Year: 2026
INN Vinča, Belgrade — OMAS Group

Standalone GP script using scikit-learn GaussianProcessRegressor.
Predicts Dq/B for new phosphor candidates with uncertainty quantification.

Usage:
    python gp_dqb_prediction.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import io

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Paths — hardcoded
# ─────────────────────────────────────────────
TRAIN_PATH   = r"C:\Users\Korisnik\PycharmProjects\Gausian Processes\Cr3_dqb_training_set.xlsx"
PREDICT_PATH = r"C:\Users\Korisnik\PycharmProjects\Gausian Processes\Prošireni To predict.xlsx"
OUTPUT_PATH  = r"C:\Users\Korisnik\PycharmProjects\Gausian Processes\gp_dqb_results.xlsx"
PLOT_PRED    = r"C:\Users\Korisnik\PycharmProjects\Gausian Processes\gp_predictions_plot.png"
PLOT_ARD     = r"C:\Users\Korisnik\PycharmProjects\Gausian Processes\gp_ard_relevance.png"

# ─────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────
TARGET_DQB_MIN = 2.2
TARGET_DQB_MAX = 2.8
N_RESTARTS     = 5
SEED           = 42
np.random.seed(SEED)


# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────

def load_training_data(filepath: str):
    """
    Training file format:
        Column 0   : Formula (string)
        Column 1   : Dq/B (target)
        Columns 2..: Features
    """
    df = pd.read_excel(filepath)
    feature_names = list(df.columns[2:])
    X = df.iloc[:, 2:].values.astype(np.float64)
    y = df.iloc[:, 1].values.astype(np.float64)
    print(f"[Data] Training samples: {X.shape[0]}, Features: {X.shape[1]}")
    return X, y, feature_names


def load_prediction_data(filepath: str):
    """
    Prediction file format (no header):
        Column 0   : Formula (string)
        Columns 1..: Features (same order as training)
    """
    df = pd.read_excel(filepath, header=None)
    formulas = df.iloc[:, 0].values.astype(str)
    X = df.iloc[:, 1:].values.astype(np.float64)
    print(f"[Data] Prediction candidates: {X.shape[0]}")
    return formulas, X


# ─────────────────────────────────────────────
# GP Model
# ─────────────────────────────────────────────

class GPDqBPredictor:
    """
    Gaussian Process model for Dq/B prediction.
    Uses scikit-learn GPR with ARD Matérn 5/2 kernel.
    RobustScaler handles extreme feature distributions.
    """

    def __init__(self, n_restarts: int = 5):
        self.n_restarts = n_restarts
        self.scaler_X = RobustScaler()
        self.model = None
        self.feature_names = None

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names=None):
        self.feature_names = feature_names
        n_features = X.shape[1]

        X_scaled = self.scaler_X.fit_transform(X)

        # ARD Matérn 5/2 kernel + noise
        kernel = (
            ConstantKernel(1.0, constant_value_bounds=(1e-3, 10.0)) *
            Matern(
                length_scale=np.ones(n_features),
                length_scale_bounds=(0.01, 100.0),
                nu=2.5
            ) +
            WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5, 1.0))
        )

        self.model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=self.n_restarts,
            normalize_y=True,
            alpha=1e-6,
            random_state=SEED
        )

        print(f"  [GP] Fitting with {self.n_restarts} optimizer restarts...")
        self.model.fit(X_scaled, y)
        print(f"  [GP] Log-marginal-likelihood: {self.model.log_marginal_likelihood_value_:.4f}")
        return self

    def predict(self, X: np.ndarray):
        X_scaled = self.scaler_X.transform(X)
        mean, std = self.model.predict(X_scaled, return_std=True)
        lower_95 = mean - 1.96 * std
        upper_95 = mean + 1.96 * std
        return mean, std, lower_95, upper_95

    def get_length_scales(self):
        # Extract ARD length-scales from fitted kernel
        k = self.model.kernel_
        # kernel structure: ConstantKernel * Matern + WhiteKernel
        matern = k.k1.k2
        return matern.length_scale

    def print_hyperparameters(self):
        print("\n[GP] Optimized kernel:")
        print(f"  {self.model.kernel_}")
        ls = self.get_length_scales()
        print(f"\n  Length-scales (per feature):")
        if self.feature_names:
            for name, l in zip(self.feature_names, ls):
                print(f"    {name:45s}: {l:.4f}  (relevance: {1/l:.4f})")
        else:
            print(f"  {ls}")


# ─────────────────────────────────────────────
# Cross-Validation
# ─────────────────────────────────────────────

def cross_validate_gp(X, y, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    rmse_scores, r2_scores, calib_scores = [], [], []

    print(f"\n[CV] Running {n_folds}-fold cross-validation...")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        gp = GPDqBPredictor(n_restarts=3)
        gp.fit(X_tr, y_tr)
        mean, std, _, _ = gp.predict(X_val)

        rmse = np.sqrt(mean_squared_error(y_val, mean))
        r2   = r2_score(y_val, mean)
        calib = np.mean(np.abs(y_val - mean) <= std)

        rmse_scores.append(rmse)
        r2_scores.append(r2)
        calib_scores.append(calib)
        print(f"  Fold {fold+1}: RMSE={rmse:.4f}, R²={r2:.4f}, Within-1σ={calib:.2%}")

    print(f"\n[CV] Summary:")
    print(f"  RMSE : {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
    print(f"  R²   : {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
    print(f"  Calib: {np.mean(calib_scores):.2%} ± {np.std(calib_scores):.2%} (expected ~68%)")


# ─────────────────────────────────────────────
# Tier Classification
# ─────────────────────────────────────────────

def classify_tier(mean, std, target_min, target_max):
    in_target  = (mean >= target_min) & (mean <= target_max)
    near_target = (
        ((mean >= target_min - 0.3) & (mean < target_min)) |
        ((mean > target_max) & (mean <= target_max + 0.3))
    )
    tiers = []
    for m, s, it, nt in zip(mean, std, in_target, near_target):
        if it and s < 0.2:
            tiers.append("Tier 1 — Strong")
        elif it and s < 0.4:
            tiers.append("Tier 2 — Promising")
        elif it:
            tiers.append("Tier 3 — Uncertain")
        elif nt:
            tiers.append("Tier 3 — Edge")
        else:
            tiers.append("Tier 4 — Out of range")
    return tiers, in_target


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────

def plot_predictions(formulas, mean, std, lower_95, upper_95,
                     target_range, save_path):
    idx = np.argsort(mean)[::-1]
    fig, ax = plt.subplots(figsize=(12, 7))
    x_pos = np.arange(len(idx))

    ax.errorbar(x_pos, mean[idx], yerr=1.96 * std[idx],
                fmt='o', color='#2196F3', ecolor='#90CAF9',
                capsize=4, markersize=6, label='Predicted Dq/B ± 95% CI')

    ax.axhspan(target_range[0], target_range[1],
               alpha=0.15, color='green',
               label=f'Target range [{target_range[0]}, {target_range[1]}]')
    ax.axhline(target_range[0], color='green', linestyle='--', linewidth=0.8)
    ax.axhline(target_range[1], color='green', linestyle='--', linewidth=0.8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [f[:14] + '…' if len(f) > 14 else f for f in formulas[idx]],
        rotation=45, ha='right', fontsize=9
    )
    ax.set_ylabel("Predicted Dq/B", fontsize=12)
    ax.set_title("GP Predictions: Dq/B for Candidates", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[Plot] Saved to: {save_path}")
    plt.close()


def plot_ard_relevance(gp: GPDqBPredictor, save_path: str):
    ls = gp.get_length_scales()
    relevance = 1.0 / ls
    n = len(ls)
    feature_names = gp.feature_names if gp.feature_names else [f"Feature {i+1}" for i in range(n)]

    idx = np.argsort(relevance)[::-1]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(n), relevance[idx], color='#4CAF50', alpha=0.8)
    ax.set_xticks(range(n))
    ax.set_xticklabels([feature_names[i] for i in idx],
                       rotation=45, ha='right', fontsize=9)
    ax.set_ylabel("Relevance (1 / length-scale)", fontsize=12)
    ax.set_title("ARD Feature Relevance (GP Kernel)", fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[Plot] Saved to: {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# Save Results
# ─────────────────────────────────────────────

def save_results(formulas, mean, std, lower_95, upper_95, tiers, in_target, output_path):
    df = pd.DataFrame({
        "Formula":              formulas,
        "GP_Predicted_DqB":     np.round(mean, 4),
        "GP_Uncertainty_1sigma":np.round(std, 4),
        "GP_Lower_95CI":        np.round(lower_95, 4),
        "GP_Upper_95CI":        np.round(upper_95, 4),
        "In_Target_Range":      in_target,
        "Tier":                 tiers
    })
    df_sorted = df.sort_values("GP_Predicted_DqB", ascending=False)
    df_sorted.to_excel(output_path, index=False)
    print(f"[Output] Results saved to: {output_path}")

    print(f"\n[Summary] Target range: [{TARGET_DQB_MIN}, {TARGET_DQB_MAX}]")
    print(f"  Candidates in target range : {in_target.sum()}")
    for tier in ["Tier 1", "Tier 2", "Tier 3", "Tier 4"]:
        count = sum(1 for t in tiers if tier in t)
        print(f"  {tier}: {count}")
    return df_sorted


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Cr³⁺ Phosphor GP Dq/B Predictor")
    print("  Author: Snežana Đurković | INN Vinča 2026")
    print("=" * 60)

    # Load data
    X_train, y_train, feature_names = load_training_data(TRAIN_PATH)
    formulas, X_pred = load_prediction_data(PREDICT_PATH)

    # Fit GP
    print(f"\n[GP] Fitting model (Matérn 5/2 ARD, {N_RESTARTS} restarts)...")
    gp = GPDqBPredictor(n_restarts=N_RESTARTS)
    gp.fit(X_train, y_train, feature_names=feature_names)
    gp.print_hyperparameters()

    # Predict
    print("\n[GP] Predicting Dq/B for candidates...")
    mean, std, lower_95, upper_95 = gp.predict(X_pred)

    # Classify
    tiers, in_target = classify_tier(mean, std, TARGET_DQB_MIN, TARGET_DQB_MAX)

    # Save
    results = save_results(formulas, mean, std, lower_95, upper_95,
                           tiers, in_target, OUTPUT_PATH)

    # Plots
    plot_predictions(formulas, mean, std, lower_95, upper_95,
                     target_range=(TARGET_DQB_MIN, TARGET_DQB_MAX),
                     save_path=PLOT_PRED)

    plot_ard_relevance(gp, save_path=PLOT_ARD)

    print("\n[Done] GP pipeline complete.")
    return results


if __name__ == "__main__":
    main()
