# Battery cycling data loader — Severson et al. dataset
# Parses the HDF5-backed MATLAB struct (.mat, v7.3) for one batch into a
# plain nested-dict format and pickles it for downstream feature engineering

'''
Struct layout (per cell):
    cycle_life       — scalar, total cycles to end of life
    charge_policy    — string, fast-charging protocol
    summary          — per-cycle aggregates (IR, QC, QD, temps, chargetime)
    cycles           — per-cycle raw time series (I, V, T, Qc, Qd, Qdlin, Tdlin, dQdV, t)
'''

import argparse
import pickle
from pathlib import Path
import h5py
import numpy as np

def _decode_policy(f, ref):
    """Decode the MATLAB char-array policy string stored as uint16 codes."""
    raw = f[ref][()]
    return raw.tobytes()[::2].decode()


def load_batch(mat_path: str) -> dict:
    """Parse one batch .mat file into {cell_id: cell_dict}."""
    f = h5py.File(mat_path, "r")
    batch = f["batch"]
    num_cells = batch["summary"].shape[0]

    bat_dict = {}
    for i in range(num_cells):
        cycle_life = f[batch["cycle_life"][i, 0]][()]
        policy = _decode_policy(f, batch["policy_readable"][i, 0])

        s = f[batch["summary"][i, 0]]
        summary = {
            "IR": np.hstack(s["IR"][0, :].tolist()),
            "QC": np.hstack(s["QCharge"][0, :].tolist()),
            "QD": np.hstack(s["QDischarge"][0, :].tolist()),
            "Tavg": np.hstack(s["Tavg"][0, :].tolist()),
            "Tmin": np.hstack(s["Tmin"][0, :].tolist()),
            "Tmax": np.hstack(s["Tmax"][0, :].tolist()),
            "chargetime": np.hstack(s["chargetime"][0, :].tolist()),
            "cycle": np.hstack(s["cycle"][0, :].tolist()),
        }

        cycles = f[batch["cycles"][i, 0]]
        cycle_dict = {}
        for j in range(cycles["I"].shape[0]):
            cycle_dict[str(j)] = {
                "I": np.hstack(f[cycles["I"][j, 0]][()]),
                "Qc": np.hstack(f[cycles["Qc"][j, 0]][()]),
                "Qd": np.hstack(f[cycles["Qd"][j, 0]][()]),
                "Qdlin": np.hstack(f[cycles["Qdlin"][j, 0]][()]),
                "T": np.hstack(f[cycles["T"][j, 0]][()]),
                "Tdlin": np.hstack(f[cycles["Tdlin"][j, 0]][()]),
                "V": np.hstack(f[cycles["V"][j, 0]][()]),
                "dQdV": np.hstack(f[cycles["discharge_dQdV"][j, 0]][()]),
                "t": np.hstack(f[cycles["t"][j, 0]][()]),
            }

        bat_dict[f"b{i}"] = {
            "cycle_life": cycle_life,
            "charge_policy": policy,
            "summary": summary,
            "cycles": cycle_dict,
        }

    f.close()
    return bat_dict


def main():
    parser = argparse.ArgumentParser(description="Convert a Severson batch .mat file to .pkl")
    parser.add_argument("mat_path", help="Path to the batch .mat file")
    parser.add_argument("out_path", help="Path to write the output .pkl file")
    parser.add_argument("--batch-prefix", default="b1", help="Prefix for cell keys, e.g. b1c0, b2c0")
    args = parser.parse_args()

    print(f"Loading {args.mat_path} ...")
    bat_dict = load_batch(args.mat_path)

    # re-key with batch prefix so cells stay unique across batches once merged
    bat_dict = {f"{args.batch_prefix}c{k[1:]}": v for k, v in bat_dict.items()}

    print(f"Loaded {len(bat_dict)} cells. Writing to {args.out_path} ...")
    with open(args.out_path, "wb") as fp:
        pickle.dump(bat_dict, fp)
    print("Done.")


if __name__ == "__main__":
    main()