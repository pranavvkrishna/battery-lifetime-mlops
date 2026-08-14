# Train a model, log it to MLflow, and register it in the Model Registry

""" Usage:
    python mlflow_track.py rf   --features features_rul.csv
    python mlflow_track.py xgb  --features features_rul.csv
    python mlflow_track.py mlp  --features features_rul.csv
    python mlflow_track.py lstm --sequences sequences_rul.npz
    python mlflow_track.py cnn  --sequences sequences_rul.npz """

import argparse
import sys
from pathlib import Path
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.pyfunc
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "train"))
from common import load_features, run_cv

EXPERIMENT_NAME = "battery-rul"
REGISTRY_NAME = "battery-rul-model"


class MLPWrapper(mlflow.pyfunc.PythonModel):
    """Wraps the Keras MLP + its scaler/normalization so predict() always
    returns raw RUL values (cycles), matching RF/XGBoost's output — not
    the internal normalized scale the network was actually trained on."""

    def __init__(self, keras_model, scaler, y_mean, y_std):
        self.keras_model = keras_model
        self.scaler = scaler
        self.y_mean = y_mean
        self.y_std = y_std

    def predict(self, context, model_input):
        X_scaled = self.scaler.transform(model_input)
        pred_norm = self.keras_model.predict(X_scaled, verbose=0).flatten()
        return pred_norm * self.y_std + self.y_mean


def train_rf(features_path, seed):
    from sklearn.ensemble import RandomForestRegressor
    X, y, groups = load_features(features_path)
    params = {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 2, "random_state": seed}
    model_fn = lambda: RandomForestRegressor(**params)
    cv_results = run_cv(model_fn, X, y, groups, seed=seed)
    final_model = RandomForestRegressor(**params).fit(X, y)
    return final_model, params, cv_results, "sklearn", X.iloc[:5]


def train_xgb(features_path, seed):
    import xgboost as xgb
    X, y, groups = load_features(features_path)
    params = {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
              "subsample": 0.8, "colsample_bytree": 0.8, "random_state": seed}
    model_fn = lambda: xgb.XGBRegressor(**params)
    cv_results = run_cv(model_fn, X, y, groups, seed=seed)
    final_model = xgb.XGBRegressor(**params).fit(X, y)
    return final_model, params, cv_results, "xgboost", X.iloc[:5]


def train_mlp(features_path, seed):
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    X, y, groups = load_features(features_path)
    y_mean, y_std = y.mean(), y.std()
    y_norm = (y - y_mean) / y_std

    params = {"layers": "64-32-16", "learning_rate": 1e-3, "epochs": 100, "batch_size": 32}

    def build():
        m = models.Sequential([
            layers.Input(shape=(X.shape[1],)),
            layers.Dense(64, activation="relu"), layers.Dropout(0.2),
            layers.Dense(32, activation="relu"), layers.Dropout(0.2),
            layers.Dense(16, activation="relu"),
            layers.Dense(1),
        ])
        m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]), loss="huber")
        return m

    gkf = GroupKFold(n_splits=min(5, len(set(groups))))
    maes, rmses, r2s = [], [], []
    for train_idx, test_idx in gkf.split(X, y_norm, groups):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X.iloc[train_idx])
        X_test = scaler.transform(X.iloc[test_idx])
        m = build()
        m.fit(X_train, y_norm[train_idx], epochs=params["epochs"], batch_size=params["batch_size"], verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)])
        pred = m.predict(X_test, verbose=0).flatten() * y_std + y_mean
        y_true = y[test_idx]
        maes.append(mean_absolute_error(y_true, pred))
        rmses.append(np.sqrt(mean_squared_error(y_true, pred)))
        r2s.append(r2_score(y_true, pred))

    cv_results = {"MAE_mean": np.mean(maes), "MAE_std": np.std(maes),
                  "RMSE_mean": np.mean(rmses), "RMSE_std": np.std(rmses),
                  "R2_mean": np.mean(r2s), "R2_std": np.std(r2s)}

    scaler = StandardScaler().fit(X)
    final_model = build()
    final_model.fit(scaler.transform(X), y_norm, epochs=params["epochs"],
                     batch_size=params["batch_size"], verbose=0)

    wrapped_model = MLPWrapper(final_model, scaler, y_mean, y_std)
    return wrapped_model, params, cv_results, "pyfunc", X.iloc[:5]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_type", choices=["rf", "xgb", "mlp"])  # lstm/cnn use raw sequences — see note below
    parser.add_argument("--features", help="Path to features_rul.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tracking-uri", default=None, help="MLflow tracking URI (local ./mlruns if omitted)")
    args = parser.parse_args()

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    trainers = {"rf": train_rf, "xgb": train_xgb, "mlp": train_mlp}
    model, params, cv_results, flavor, sample_input = trainers[args.model_type](args.features, args.seed)

    with mlflow.start_run(run_name=args.model_type) as run:
        mlflow.log_param("model_type", args.model_type)
        for k, v in params.items():
            mlflow.log_param(k, v)
        for k, v in cv_results.items():
            mlflow.log_metric(k, v)

        if flavor == "sklearn":
            mlflow.sklearn.log_model(model, "model", registered_model_name=REGISTRY_NAME)
        elif flavor == "xgboost":
            mlflow.xgboost.log_model(model, "model", registered_model_name=REGISTRY_NAME)
        elif flavor == "pyfunc":
            mlflow.pyfunc.log_model("model", python_model=model, registered_model_name=REGISTRY_NAME)

        print(f"Logged run: {run.info.run_id}")
        print(f"MAE: {cv_results['MAE_mean']:.2f} +/- {cv_results['MAE_std']:.2f}")
        print(f"R2:  {cv_results['R2_mean']:.3f} +/- {cv_results['R2_std']:.3f}")
        print(f"Registered as '{REGISTRY_NAME}' — check 'mlflow ui' for the new version number")


if __name__ == "__main__":
    main()