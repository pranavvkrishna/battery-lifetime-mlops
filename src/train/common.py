# Shared helpers for training scripts — data loading and CV evaluation
# Used by train_rf.py, train_xgb.py, train_mlp.py, train_lstm.py and train_cnn.py

""" Uses GroupKFold (grouped by cell_id) instead of plain KFold, since
features_rul.csv has multiple rows per battery a plain split would
leak the same battery into both train and test """


import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold


def load_features(features_csv: str):
    """Load features_rul.csv, split into X (features), y (RUL target), groups (cell_id)"""
    df = pd.read_csv(features_csv)

    target_col = "rul" if "rul" in df.columns else "cycle_life"
    df = df.dropna(subset=[target_col])
    y = df[target_col].values
    groups = df["cell_id"].values

    drop_cols = ["cell_id", "charge_policy", "cycle_life", "rul", "checkpoint_cycle"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X = X.fillna(X.median(numeric_only=True))

    return X, y, groups


def run_cv(model_fn, X, y, groups, n_splits=5, seed=42):
    """Run GroupKFold CV (grouped by battery) — no cell appears in both train and test"""
    n_splits = min(n_splits, len(np.unique(groups)))  # can't have more folds than groups
    gkf = GroupKFold(n_splits=n_splits)
    maes, rmses, r2s = [], [], []

    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = model_fn()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        maes.append(mean_absolute_error(y_test, pred))
        rmses.append(np.sqrt(mean_squared_error(y_test, pred)))
        r2s.append(r2_score(y_test, pred))

    return {
        "MAE_mean": np.mean(maes), "MAE_std": np.std(maes),
        "RMSE_mean": np.mean(rmses), "RMSE_std": np.std(rmses),
        "R2_mean": np.mean(r2s), "R2_std": np.std(r2s),
    }


def print_results(model_name, r):
    print(f"{model_name:15s}  MAE: {r['MAE_mean']:7.2f} +/- {r['MAE_std']:5.2f}"
          f"   RMSE: {r['RMSE_mean']:7.2f} +/- {r['RMSE_std']:5.2f}"
          f"   R2: {r['R2_mean']:.3f} +/- {r['R2_std']:.3f}")