# Train a small MLP on features_rul.csv — same engineered features as
# RF/XGBoost, to test whether neural nets do fine here when not fed raw
# sequences like the LSTM


import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models

from common import load_features

def build_model(n_features):
    model = models.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="huber")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("features_csv", help="Path to features_rul.csv")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X, y, groups = load_features(args.features_csv)
    print(f"Loaded {len(X)} rows across {len(set(groups))} batteries, {X.shape[1]} features")

    y_mean, y_std = y.mean(), y.std()
    y_norm = (y - y_mean) / y_std

    n_splits = min(args.n_splits, len(set(groups)))
    gkf = GroupKFold(n_splits=n_splits)

    maes, rmses, r2s = [], [], []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y_norm, groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_norm[train_idx], y_norm[test_idx]

        # standardize features per fold (fit on train only, avoids leakage)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = build_model(X.shape[1])
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=10, restore_best_weights=True
        )
        model.fit(
            X_train_s, y_train,
            epochs=args.epochs, batch_size=args.batch_size,
            callbacks=[early_stop], verbose=1 if fold == 0 else 0,
        )

        pred_norm = model.predict(X_test_s, verbose=0).flatten()
        pred = pred_norm * y_std + y_mean
        y_test_actual = y_test * y_std + y_mean

        mae = mean_absolute_error(y_test_actual, pred)
        rmse = np.sqrt(mean_squared_error(y_test_actual, pred))
        r2 = r2_score(y_test_actual, pred)
        print(f"Fold {fold + 1}: MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.3f}")

        maes.append(mae)
        rmses.append(rmse)
        r2s.append(r2)

    print(f"\nMLP             MAE: {np.mean(maes):7.2f} +/- {np.std(maes):5.2f}"
          f"   RMSE: {np.mean(rmses):7.2f} +/- {np.std(rmses):5.2f}"
          f"   R2: {np.mean(r2s):.3f} +/- {np.std(r2s):.3f}")

    results = {
        "model": "MLP",
        "MAE_mean": np.mean(maes), "MAE_std": np.std(maes),
        "RMSE_mean": np.mean(rmses), "RMSE_std": np.std(rmses),
        "R2_mean": np.mean(r2s), "R2_std": np.std(r2s),
    }
    out_path = Path(args.features_csv).parent / "mlp_rul_results.csv"
    pd.DataFrame([results]).to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()