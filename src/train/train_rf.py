# Train Random Forest on features_rul.csv, grouped 5-fold CV by battery (no leakage)

import argparse
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from common import load_features, run_cv, print_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("features_csv", help="Path to features_rul.csv")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X, y, groups = load_features(args.features_csv)
    print(f"Loaded {len(X)} rows across {len(set(groups))} batteries, {X.shape[1]} features")
    print(f"Feature columns: {list(X.columns)}\n")

    model_fn = lambda: RandomForestRegressor(
        n_estimators=300, max_depth=8, min_samples_leaf=2, random_state=args.seed
    )
    results = run_cv(model_fn, X, y, groups, args.n_splits, args.seed)
    print_results("Random Forest", results)

    results["model"] = "Random Forest"
    out_path = Path(args.features_csv).parent / "rf_rul_results.csv"
    pd.DataFrame([results]).to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()