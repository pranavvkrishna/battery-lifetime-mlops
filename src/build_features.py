# Feature engineering for Remaining Useful Life (RUL) prediction

""" Slices each cell at multiple checkpoints (every CHECKPOINT_STRIDE cycles).
At each checkpoint, uses a trailing WINDOW of recent cycles to predict
RUL = cycle_life - checkpoint_cycle, turning 184 cells into 2724 rows """

import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

WINDOW = 50              # trailing cycles used as input at each checkpoint
CHECKPOINT_STRIDE = 50   # create a checkpoint every N cycles
MIN_RUL = 10             # skip checkpoints too close to end-of-life (trivial to predict)
SEQ_LEN = 1000           # Qdlin/Tdlin are already linearly interpolated to this length


def load_all_batches(data_dir: str) -> dict:
    all_cells = {}
    for batch_file in ["batch1.pkl", "batch2.pkl", "batch3.pkl", "batch4.pkl"]:
        path = Path(data_dir) / batch_file
        with open(path, "rb") as fp:
            batch_dict = pickle.load(fp)
        all_cells.update(batch_dict)
        print(f"  {batch_file}: {len(batch_dict)} cells")
    return all_cells


def slope(y, x):
    if len(y) < 2:
        return 0.0
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return 0.0
    return float(np.polyfit(x[mask], y[mask], 1)[0])


def build_tabular_row(cell_id, cell, checkpoint):
    """Aggregate stats from the WINDOW cycles ending at `checkpoint`."""
    summary = cell["summary"]
    start = max(0, checkpoint - WINDOW)
    end = checkpoint

    qd = np.asarray(summary["QD"][start:end], dtype=float)
    qc = np.asarray(summary["QC"][start:end], dtype=float)
    ir = np.asarray(summary["IR"][start:end], dtype=float)
    tavg = np.asarray(summary["Tavg"][start:end], dtype=float)
    tmax = np.asarray(summary["Tmax"][start:end], dtype=float)
    tmin = np.asarray(summary["Tmin"][start:end], dtype=float)
    chargetime = np.asarray(summary["chargetime"][start:end], dtype=float)
    idx = np.arange(len(qd))

    return {
        "cell_id": cell_id,
        "checkpoint_cycle": checkpoint,
        "charge_policy": cell["charge_policy"],

        "qd_current": float(qd[-1]) if len(qd) else np.nan,
        "qd_slope": slope(qd, idx),
        "qd_min": float(np.nanmin(qd)) if len(qd) else np.nan,
        "qd_std": float(np.nanstd(qd)) if len(qd) else np.nan,

        "qc_slope": slope(qc, idx),
        "qc_mean": float(np.nanmean(qc)) if len(qc) else np.nan,

        "ir_current": float(ir[-1]) if len(ir) else np.nan,
        "ir_slope": slope(ir, idx),
        "ir_mean": float(np.nanmean(ir)) if len(ir) else np.nan,

        "tavg_mean": float(np.nanmean(tavg)) if len(tavg) else np.nan,
        "tavg_std": float(np.nanstd(tavg)) if len(tavg) else np.nan,
        "tmax_mean": float(np.nanmean(tmax)) if len(tmax) else np.nan,
        "tmin_mean": float(np.nanmean(tmin)) if len(tmin) else np.nan,

        "chargetime_mean": float(np.nanmean(chargetime)) if len(chargetime) else np.nan,
        "chargetime_slope": slope(chargetime, idx),

        "window_size": len(qd),
    }


def build_sequence_window(cell, checkpoint):
    """Stack Qdlin/Tdlin for the trailing WINDOW cycles ending at checkpoint."""
    cycles = cell["cycles"]
    start = max(0, checkpoint - WINDOW)
    end = checkpoint

    seq = np.zeros((WINDOW, SEQ_LEN, 2), dtype=np.float32)
    for offset, j in enumerate(range(start, end)):
        c = cycles.get(str(j))
        if c is None:
            continue
        qdlin = np.asarray(c["Qdlin"], dtype=float)
        tdlin = np.asarray(c["Tdlin"], dtype=float)
        qdlin = qdlin[:SEQ_LEN] if len(qdlin) >= SEQ_LEN else np.pad(qdlin, (0, SEQ_LEN - len(qdlin)))
        tdlin = tdlin[:SEQ_LEN] if len(tdlin) >= SEQ_LEN else np.pad(tdlin, (0, SEQ_LEN - len(tdlin)))
        seq[offset, :, 0] = qdlin
        seq[offset, :, 1] = tdlin

    return seq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Directory with batch1.pkl, batch2.pkl, batch3.pkl")
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()

    print("Loading batches...")
    all_cells = load_all_batches(args.data_dir)
    print(f"Total cells: {len(all_cells)}")

    rows = []
    seq_list, seq_meta = [], []

    for cell_id, cell in all_cells.items():
        cycle_life = float(np.asarray(cell["cycle_life"]).squeeze())
        if np.isnan(cycle_life):
            continue  # censored cell, no defined end-of-life — skip

        n_cycles_present = len(cell["summary"]["cycle"])
        checkpoints = range(WINDOW, min(n_cycles_present, int(cycle_life)), CHECKPOINT_STRIDE)

        for checkpoint in checkpoints:
            rul = cycle_life - checkpoint
            if rul < MIN_RUL:
                continue

            try:
                row = build_tabular_row(cell_id, cell, checkpoint)
                row["cycle_life"] = cycle_life
                row["rul"] = rul
                rows.append(row)

                seq = build_sequence_window(cell, checkpoint)
                seq_list.append(seq)
                seq_meta.append((cell_id, checkpoint, rul))
            except Exception as e:
                print(f"  skipping {cell_id}@{checkpoint}: {e}")

    df = pd.DataFrame(rows)
    out_csv = Path(args.out_dir) / "features_rul.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(df)} rows (from {len(all_cells)} cells)")

    sequences = np.stack(seq_list, axis=0)
    labels = np.array([m[2] for m in seq_meta], dtype=np.float32)
    cell_ids = np.array([m[0] for m in seq_meta])
    checkpoints_arr = np.array([m[1] for m in seq_meta])

    out_npz = Path(args.out_dir) / "sequences_rul.npz"
    np.savez_compressed(
        out_npz, sequences=sequences, labels=labels,
        cell_ids=cell_ids, checkpoints=checkpoints_arr,
    )
    print(f"Wrote {out_npz} with shape {sequences.shape}")


if __name__ == "__main__":
    main()