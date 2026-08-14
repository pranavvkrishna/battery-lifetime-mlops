# Promotion logic — re-evaluates every registered model on a fixed, shared
# holdout set (cached in holdout_battery_ids.txt) instead of trusting each
# version's self-reported training-time MAE, so comparisons stay fair as new data is added

""" Usage:
    python promote.py --features features_rul.csv """

import argparse
from pathlib import Path
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error

REGISTRY_NAME = "battery-rul-model"
HOLDOUT_FRACTION = 0.2
HOLDOUT_SEED = 7  # fixed on purpose — must stay constant across runs


def get_or_create_holdout(features_csv: str, holdout_file: str) -> list:
    """Return a fixed list of battery cell_ids to hold out for evaluation.
    Cached to disk so the same batteries are used every time, even as
    new checkpoints/batteries are added to the training data."""
    holdout_path = Path(holdout_file)
    if holdout_path.exists():
        return holdout_path.read_text().strip().split("\n")

    df = pd.read_csv(features_csv)
    all_batteries = sorted(df["cell_id"].unique())
    rng = np.random.RandomState(HOLDOUT_SEED)
    n_holdout = max(1, int(len(all_batteries) * HOLDOUT_FRACTION))
    holdout = sorted(rng.choice(all_batteries, size=n_holdout, replace=False))

    holdout_path.write_text("\n".join(holdout))
    print(f"Created new holdout set: {len(holdout)} batteries -> {holdout_file}")
    return holdout


def evaluate_version(client, version, X_holdout, y_holdout):
    """Load a registered model version and score it on the shared holdout."""
    model_uri = f"models:/{REGISTRY_NAME}/{version.version}"
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        pred = np.array(model.predict(X_holdout)).flatten()
        return mean_absolute_error(y_holdout, pred)
    except Exception as e:
        print(f"  Could not evaluate version {version.version}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Path to features_rul.csv")
    parser.add_argument("--holdout-file", default="holdout_battery_ids.txt")
    args = parser.parse_args()

    holdout_ids = get_or_create_holdout(args.features, args.holdout_file)
    print(f"Evaluating all candidates on {len(holdout_ids)} held-out batteries")

    df = pd.read_csv(args.features)
    holdout_df = df[df["cell_id"].isin(holdout_ids)]
    drop_cols = ["cell_id", "charge_policy", "cycle_life", "rul", "checkpoint_cycle"]
    X_holdout = holdout_df.drop(columns=[c for c in drop_cols if c in holdout_df.columns])
    X_holdout = X_holdout.fillna(X_holdout.median(numeric_only=True))
    y_holdout = holdout_df["rul"].values

    client = MlflowClient()
    all_versions = client.search_model_versions(f"name='{REGISTRY_NAME}'")
    if not all_versions:
        print(f"No versions found for registered model '{REGISTRY_NAME}'")
        return

    print("\nRe-evaluating every version on the shared holdout set:")
    scored = []
    for v in all_versions:
        mae = evaluate_version(client, v, X_holdout, y_holdout)
        if mae is not None:
            print(f"  version {v.version} ({v.current_stage}): MAE={mae:.2f}")
            scored.append((v, mae))

    if not scored:
        print("No versions could be evaluated — nothing to promote.")
        return

    best_version, best_mae = min(scored, key=lambda pair: pair[1])
    current_prod = [v for v, _ in scored if v.current_stage == "Production"]
    current_prod_mae = next((mae for v, mae in scored if v.current_stage == "Production"), None)

    print(f"\nBest on holdout: version {best_version.version} (MAE {best_mae:.2f})")
    if current_prod:
        print(f"Current production: version {current_prod[0].version} (MAE {current_prod_mae:.2f})")
    else:
        print("Current production: none")

    if best_version.current_stage == "Production":
        print("No change — current production is already the best model on the holdout set.")
        return

    client.transition_model_version_stage(
        name=REGISTRY_NAME, version=best_version.version, stage="Production",
        archive_existing_versions=True,
    )
    print(f"PROMOTED: version {best_version.version} is now Production "
          f"(MAE {best_mae:.2f} on shared holdout)")


if __name__ == "__main__":
    main()