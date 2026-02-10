#%%
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

X_GROUPS = [f"carbonate_x{i:02d}" for i in range(1, 31)]
T_LEVELS = ["0.00", "0.10", "0.20", "0.50", "1.00"]
DS_NAMES = [f"t_{t}" for t in T_LEVELS]


def read_h5_nested(filepath: str | Path,
                   groups: List[str] = X_GROUPS,
                   datasets: List[str] = DS_NAMES) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Read an HDF5 file with structure:
      /carbonate_xXX/t_0.00, t_0.10, ...
    Returns: data[group][dataset] = np.ndarray
    """
    filepath = Path(filepath)
    out: Dict[str, Dict[str, np.ndarray]] = {}

    with h5py.File(filepath, "r") as f:
        for g in groups:
            if g not in f:
                continue
            out[g] = {}
            grp = f[g]
            for d in datasets:
                if d in grp:
                    out[g][d] = grp[d][()]  # read dataset into numpy array (or scalar)
    return out


def read_all(prefix: str, name: str, base_dir: str | Path = ".") -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """
    Reads adv_t_, conc_, pl_ files for a given <name>.
    Returns:
      result['adv_t'][group][dataset]
      result['conc'][group][dataset]
      result['pl'][group][dataset]
    """
    base_dir = Path(base_dir)

    files = {
        "adv_t": base_dir / f"adv_t_{name}",
        "conc":  base_dir / f"conc_{name}",
        "pl":    base_dir / f"pl_{name}",
    }

    result = {}
    for key, fp in files.items():
        if not fp.exists():
            raise FileNotFoundError(f"Missing file: {fp}")
        result[key] = read_h5_nested(fp)

    return result


def to_stacked_array(nested: Dict[str, Dict[str, np.ndarray]],
                     groups: List[str] = X_GROUPS,
                     datasets: List[str] = DS_NAMES
                     ) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Converts nested dict (one file) into a stacked object array of shape (n_groups, n_datasets),
    where each entry can be a scalar or 1D/2D array depending on your stored data rows.
    """
    arr = np.empty((len(groups), len(datasets)), dtype=object)
    for i, g in enumerate(groups):
        for j, d in enumerate(datasets):
            arr[i, j] = nested.get(g, {}).get(d, None)
    return arr, groups, datasets

#%%
# --- Example usage ---

name = "G0_10_Daeff0_0002.hdf5"

data = read_all(prefix="", name=name, base_dir=".")
# Access examples:
# adv time data for carbonate_x03 at t=0.20:
adv_x03_t020 = data["adv_t"]["carbonate_x03"]["t_0.20"]
print("adv_x03_t020:", adv_x03_t020)

# Convert one file to a 30x5 stacked array:
adv_arr, groups, dsets = to_stacked_array(data["adv_t"])
print("adv_arr shape:", adv_arr.shape)  # (30, 5)

# %%
