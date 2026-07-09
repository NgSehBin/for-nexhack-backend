"""
ml/train.py — XGBoost model training pipeline.

Usage:
  python ml/train.py                          # train on ml/artifacts/features.csv
  python ml/train.py --features path/to.csv  # custom feature file

Steps:
  1. Load features.csv (produced by feature_pipeline.py)
  2. Train XGBoost with 5-fold stratified cross-validation
  3. Calibrate probabilities with isotonic regression
  4. Save model + feature list to ml/artifacts/
  5. Generate SHAP summary plot
  6. Print precision/recall/F1 at threshold 0.5

Outputs:
  ml/artifacts/mule_scorer.pkl    — trained calibrated model
  ml/artifacts/feature_names.pkl  — ordered feature list
  ml/artifacts/shap_summary.png   — SHAP beeswarm plot for demo slides
"""
import argparse
import sys
from pathlib import Path


FEATURE_COLS = [
    "fan_in_count",
    "fan_in_amount",
    "fan_out_count",
    "fan_out_amount",
    "pass_through_ratio",
    "avg_pass_minutes",
    "in_degree",
    "out_degree",
    "degree_ratio",
    "in_out_amount_diff",
    "proximity_to_mule",
    "known_mule",
]
LABEL_COL = "is_mule"
ARTIFACTS_DIR = Path("ml/artifacts")


def train(features_csv: str = "ml/artifacts/features.csv"):
    import joblib
    import numpy as np
    import pandas as pd
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import (
        classification_report,
        precision_recall_curve,
        roc_auc_score,
        average_precision_score,
    )
    from sklearn.model_selection import StratifiedKFold
    from xgboost import XGBClassifier

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    print(f"Loading features from {features_csv}...")
    df = pd.read_csv(features_csv)

    # Fill missing feature columns with 0
    for col in FEATURE_COLS:
        if col not in df.columns:
            print(f"  Warning: feature '{col}' missing — filling with 0")
            df[col] = 0

    X = df[FEATURE_COLS].fillna(0)
    y = df[LABEL_COL].astype(int)

    print(f"  Rows: {len(df)}, Mule: {y.sum()}, Clean: {(y==0).sum()}")

    if y.sum() < 3:
        print("ERROR: Not enough mule samples (need ≥3) — run data_gen.py with more data.")
        sys.exit(1)

    # ── Class imbalance weight ─────────────────────────────────────────────
    neg, pos = (y == 0).sum(), (y == 1).sum()
    scale_pos_weight = neg / max(pos, 1)
    print(f"  scale_pos_weight: {scale_pos_weight:.2f}")

    # ── Base XGBoost classifier ────────────────────────────────────────────
    base_model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",        # area under PR curve — better for imbalanced data
        use_label_encoder=False,
        random_state=42,
        verbosity=0,
    )

    # ── 5-fold stratified CV ───────────────────────────────────────────────
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_roc_scores = []
    auc_pr_scores = []

    print("\nCross-validation:")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        fold_model = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr", use_label_encoder=False,
            random_state=42, verbosity=0,
        )
        fold_model.fit(X_tr, y_tr)
        preds = fold_model.predict_proba(X_val)[:, 1]

        roc = roc_auc_score(y_val, preds) if y_val.sum() > 0 else 0
        pr = average_precision_score(y_val, preds) if y_val.sum() > 0 else 0
        auc_roc_scores.append(roc)
        auc_pr_scores.append(pr)
        print(f"  Fold {fold+1}: AUC-ROC={roc:.4f}  AUC-PR={pr:.4f}")

    print(f"\nMean AUC-ROC: {np.mean(auc_roc_scores):.4f} ± {np.std(auc_roc_scores):.4f}")
    print(f"Mean AUC-PR:  {np.mean(auc_pr_scores):.4f} ± {np.std(auc_pr_scores):.4f}")

    # ── Train final model on ALL data with probability calibration ─────────
    print("\nTraining final model on all data...")
    calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv=5)
    calibrated.fit(X, y)

    # ── Evaluation on full training set (optimistic but useful for demo) ───
    y_pred_proba = calibrated.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    print("\nClassification report (training set, threshold=0.5):")
    print(classification_report(y, y_pred, target_names=["Clean", "Mule"], zero_division=0))

    # ── Save artifacts ─────────────────────────────────────────────────────
    model_path = ARTIFACTS_DIR / "mule_scorer.pkl"
    features_path = ARTIFACTS_DIR / "feature_names.pkl"
    joblib.dump(calibrated, model_path)
    joblib.dump(FEATURE_COLS, features_path)
    print(f"Model saved:    {model_path}")
    print(f"Features saved: {features_path}")

    # ── SHAP summary plot ──────────────────────────────────────────────────
    print("\nGenerating SHAP summary plot...")
    try:
        # Use the underlying XGBoost estimator (not the calibration wrapper)
        underlying = calibrated.calibrated_classifiers_[0].estimator
        explainer = shap.TreeExplainer(underlying)
        shap_values = explainer.shap_values(X)

        plt.figure(figsize=(9, 5))
        shap.summary_plot(shap_values, X, feature_names=FEATURE_COLS,
                          show=False, plot_size=None)
        plt.title("SHAP feature importance — mule risk model", fontsize=12, pad=10)
        plt.tight_layout()
        shap_path = ARTIFACTS_DIR / "shap_summary.png"
        plt.savefig(shap_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"SHAP plot saved: {shap_path}")
    except Exception as e:
        print(f"  SHAP plot failed (non-fatal): {e}")

    print("\nTraining complete.")
    return calibrated


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="ml/artifacts/features.csv",
                        help="Path to features CSV")
    args = parser.parse_args()

    if not Path(args.features).exists():
        print(f"Feature file not found: {args.features}")
        print("Run: python ml/feature_pipeline.py")
        sys.exit(1)

    train(features_csv=args.features)