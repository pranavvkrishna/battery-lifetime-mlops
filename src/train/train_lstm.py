# Train an LSTM on sequences_rul.npz — trailing 50-cycle windows predicting
# Remaining Useful Life at each checkpoint

""" Uses GroupKFold by cell_id to prevent battery leakage. Includes stability
fixes (clipping, downsampling, normalization, gradient clipping) to avoid
the NaN-loss divergence hit in earlier versions """

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(n_cycles, seq_len, n_channels):
    model = models.Sequential([
        layers.Input(shape=(n_cycles, seq_len, n_channels)),
        layers.Reshape((n_cycles, seq_len * n_channels)),
        layers.LayerNormalization(),
        layers.Masking(mask_value=0.0),
        layers.LSTM(32, return_sequences=True, dropout=0.3),
        layers.LSTM(16, dropout=0.3),
        layers.Dense(8, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1),
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss="huber")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sequences_npz", help="Path to sequences_rul.npz")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = np.load(args.sequences_npz)
    X = data["sequences"]
    y = data["labels"].astype(np.float32)
    groups = data["cell_ids"]
    print(f"Loaded sequences: {X.shape}, labels: {y.shape}, batteries: {len(set(groups))}")

    n_nan = np.isnan(X).sum()
    print(f"NaN values: {n_nan}")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    lo, hi = np.percentile(X[X != 0], [0.5, 99.5]) if (X != 0).any() else (0, 1)
    X = np.clip(X, lo, hi)
    print(f"Clipped X range: [{lo:.3f}, {hi:.3f}]")

    # downsample 1000 -> 100 points per cycle to shrink input dimensionality
    X = X[:, :, ::10, :]
    print(f"Downsampled sequence shape: {X.shape}")

    X_mean, X_std = X.mean(), X.std() + 1e-8
    X_norm = (X - X_mean) / X_std

    y_mean, y_std = y.mean(), y.std()
    y_norm = (y - y_mean) / y_std
    print(f"Label (RUL) range: min={y.min():.1f}, max={y.max():.1f}, mean={y_mean:.1f}, std={y_std:.1f}")

    n_cycles, seq_len, n_channels = X.shape[1], X.shape[2], X.shape[3]
    n_splits = min(args.n_splits, len(set(groups)))
    gkf = GroupKFold(n_splits=n_splits)

    maes, rmses, r2s = [], [], []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_norm, y_norm, groups)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")
        X_train, X_test = X_norm[train_idx], X_norm[test_idx]
        y_train, y_test = y_norm[train_idx], y_norm[test_idx]

        model = build_model(n_cycles, seq_len, n_channels)
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=6, restore_best_weights=True
        )
        terminate_nan = tf.keras.callbacks.TerminateOnNaN()
        history = model.fit(
            X_train, y_train,
            epochs=args.epochs, batch_size=args.batch_size,
            callbacks=[early_stop, terminate_nan], verbose=1 if fold == 0 else 0,
        )
        if fold == 0:
            print(f"Loss history (fold 1): {history.history['loss'][-5:]}")

        pred_norm = model.predict(X_test, verbose=0).flatten()
        if np.isnan(pred_norm).any():
            print(f"Fold {fold + 1}: model produced NaN predictions — skipping fold")
            continue

        pred = pred_norm * y_std + y_mean
        y_test_actual = y_test * y_std + y_mean

        mae = mean_absolute_error(y_test_actual, pred)
        rmse = np.sqrt(mean_squared_error(y_test_actual, pred))
        r2 = r2_score(y_test_actual, pred)
        print(f"Fold {fold + 1}: MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.3f}")

        maes.append(mae)
        rmses.append(rmse)
        r2s.append(r2)

    print(f"\nLSTM            MAE: {np.mean(maes):7.2f} +/- {np.std(maes):5.2f}"
          f"   RMSE: {np.mean(rmses):7.2f} +/- {np.std(rmses):5.2f}"
          f"   R2: {np.mean(r2s):.3f} +/- {np.std(r2s):.3f}")

    results = {
        "model": "LSTM",
        "MAE_mean": np.mean(maes), "MAE_std": np.std(maes),
        "RMSE_mean": np.mean(rmses), "RMSE_std": np.std(rmses),
        "R2_mean": np.mean(r2s), "R2_std": np.std(r2s),
    }
    out_path = Path(args.sequences_npz).parent / "lstm_rul_results.csv"
    pd.DataFrame([results]).to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()