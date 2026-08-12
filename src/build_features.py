"""
Feature engineering for battery lifetime prediction.

Reads batch1.pkl, batch2.pkl, batch3.pkl (produced by load_data.py) and builds:
  1. features.csv   — one row per cell, aggregated stats from first N cycles (for RF/XGBoost)
  2. sequences.npz   — one array per cell, raw per-cycle curves for first N cycles (for LSTM)

Only the first N_CYCLES cycles are used for features — later cycles would leak
information about how long the battery actually lasted.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

N_CYCLES = 50          # only use early cycles — this is the whole point of the project
SEQ_LEN = 1000          # Qdlin/Tdlin are already linearly interpolated to a fixed length


def load_all_batches(data_dir: str) -> dict:
    """Merge batch1/2/3 pkl files into one dict keyed by cell id."""
    all_cells = {}
    for batch_file in ["batch1.pkl", "batch2.pkl", "batch3.pkl"]:
        path = Path(data_dir) / batch_file
        with open(path, "rb") as fp:
            batch_dict = pickle.load(fp)
        all_cells.update(batch_dict)
        print(f"  {batch_file}: {len(batch_dict)} cells")
    return all_cells


def build_tabular_features(cell_id: str, cell: dict) -> dict:
    """Aggregate stats from the first N_CYCLES cycles for one cell."""
    summary = cell["summary"]
    n_cycles_present = len(summary["cycle"])
    n = min(N_CYCLES, n_cycles_present)

    if n < 5:
        return None  # not enough early-cycle data to be useful

    qd = np.asarray(summary["QD"][:n], dtype=float)
    qc = np.asarray(summary["QC"][:n], dtype=float)
    ir = np.asarray(summary["IR"][:n], dtype=float)
    tavg = np.asarray(summary["Tavg"][:n], dtype=float)
    tmax = np.asarray(summary["Tmax"][:n], dtype=float)
    tmin = np.asarray(summary["Tmin"][:n], dtype=float)
    chargetime = np.asarray(summary["chargetime"][:n], dtype=float)
    cycle_idx = np.arange(n)

    def slope(y):
        # linear fit slope — crude but standard "fade rate" feature
        if len(y) < 2 or np.all(np.isnan(y)):
            return 0.0
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return 0.0
        return float(np.polyfit(cycle_idx[mask], y[mask], 1)[0])

    features = {
        "cell_id": cell_id,
        "charge_policy": cell["charge_policy"],
        "cycle_life": float(np.asarray(cell["cycle_life"]).squeeze()),

        "qd_start": float(qd[0]) if len(qd) else np.nan,
        "qd_end": float(qd[-1]) if len(qd) else np.nan,
        "qd_slope": slope(qd),
        "qd_min": float(np.nanmin(qd)) if len(qd) else np.nan,
        "qd_std": float(np.nanstd(qd)) if len(qd) else np.nan,

        "qc_slope": slope(qc),
        "qc_mean": float(np.nanmean(qc)) if len(qc) else np.nan,

        "ir_start": float(ir[0]) if len(ir) else np.nan,
        "ir_slope": slope(ir),
        "ir_mean": float(np.nanmean(ir)) if len(ir) else np.nan,

        "tavg_mean": float(np.nanmean(tavg)) if len(tavg) else np.nan,
        "tavg_std": float(np.nanstd(tavg)) if len(tavg) else np.nan,
        "tmax_mean": float(np.nanmean(tmax)) if len(tmax) else np.nan,
        "tmin_mean": float(np.nanmean(tmin)) if len(tmin) else np.nan,

        "chargetime_mean": float(np.nanmean(chargetime)) if len(chargetime) else np.nan,
        "chargetime_slope": slope(chargetime),

        "n_cycles_used": n,
    }
    return features


def build_sequence(cell_id: str, cell: dict) -> np.ndarray | None:
    """Stack Qdlin/Tdlin curves for the first N_CYCLES cycles into a (N_CYCLES, SEQ_LEN, 2) array."""
    cycles = cell["cycles"]
    n_cycles_present = len(cycles)
    n = min(N_CYCLES, n_cycles_present)
    if n < 5:
        return None

    seq = np.zeros((N_CYCLES, SEQ_LEN, 2), dtype=np.float32)  # pad with zeros if fewer than N_CYCLES
    for j in range(n):
        c = cycles[str(j)]
        qdlin = np.asarray(c["Qdlin"], dtype=float)
        tdlin = np.asarray(c["Tdlin"], dtype=float)

        # defend against length mismatches — truncate/pad to SEQ_LEN
        qdlin = qdlin[:SEQ_LEN] if len(qdlin) >= SEQ_LEN else np.pad(qdlin, (0, SEQ_LEN - len(qdlin)))
        tdlin = tdlin[:SEQ_LEN] if len(tdlin) >= SEQ_LEN else np.pad(tdlin, (0, SEQ_LEN - len(tdlin)))

        seq[j, :, 0] = qdlin
        seq[j, :, 1] = tdlin

    return seq


def main():
    parser = argparse.ArgumentParser(description="Build features.csv and sequences.npz from batch pkl files")
    parser.add_argument("data_dir", help="Directory containing batch1.pkl, batch2.pkl, batch3.pkl")
    parser.add_argument("--out-dir", default=".", help="Where to write features.csv and sequences.npz")
    args = parser.parse_args()

    print("Loading batches...")
    all_cells = load_all_batches(args.data_dir)
    print(f"Total cells: {len(all_cells)}")

    print("Building tabular features...")
    rows = []
    for cell_id, cell in all_cells.items():
        try:
            feat = build_tabular_features(cell_id, cell)
            if feat is not None:
                rows.append(feat)
        except Exception as e:
            print(f"  skipping {cell_id}: {e}")

    df = pd.DataFrame(rows)
    out_csv = Path(args.out_dir) / "features.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(df)} rows")

    print("Building sequence arrays (this is slower)...")
    seq_list = []
    seq_ids = []
    seq_labels = []
    for cell_id, cell in all_cells.items():
        try:
            seq = build_sequence(cell_id, cell)
            if seq is not None:
                seq_list.append(seq)
                seq_ids.append(cell_id)
                seq_labels.append(float(np.asarray(cell["cycle_life"]).squeeze()))
        except Exception as e:
            print(f"  skipping {cell_id}: {e}")

    sequences = np.stack(seq_list, axis=0)  # (num_cells, N_CYCLES, SEQ_LEN, 2)
    labels = np.array(seq_labels, dtype=np.float32)
    out_npz = Path(args.out_dir) / "sequences.npz"
    np.savez_compressed(out_npz, sequences=sequences, labels=labels, cell_ids=np.array(seq_ids))
    print(f"Wrote {out_npz} with shape {sequences.shape}")


if __name__ == "__main__":
    main()