import h5py
import numpy as np
from pathlib import Path

# -------------------------------------------------
#  user-configurable path
# -------------------------------------------------
h5_path = Path(r"C:\Users\abazzi\Desktop\graph-physics\dataset\train.h5")

expected_features = ["cells", "node_type", "mesh_pos", "world_pos"]

# -------------------------------------------------
#  helper functions
# -------------------------------------------------
def describe_dataset(filename: Path):
    if not filename.exists():
        raise FileNotFoundError(filename)

    with h5py.File(filename, "r") as f:
        traj_ids = list(f.keys())
        print(f"Found {len(traj_ids)} trajectories in {filename.name}\n")

        # loop over a single representative trajectory (the first one)
        traj = f[traj_ids[0]]
        print(f"⇢ Keys under trajectory '{traj_ids[0]}': {list(traj.keys())}\n")

        for feat in expected_features:
            if feat not in traj:
                print(f"⚠️  Feature '{feat}' missing in the file!")
                continue

            dset = traj[feat]
            shape = dset.shape          # (T, …)
            dtype = dset.dtype

            # grab the first time-step only to keep output short
            first_ts = dset[0]
            print(f"{feat:>10}: shape {shape}, dtype {dtype}")

            if feat == "node_type":
                uniques = np.unique(first_ts)
                print(f"  ↳ unique node_type values (t=0): {uniques}")
        print("\nDone.")

# -------------------------------------------------
#  run it
# -------------------------------------------------
if __name__ == "__main__":
    describe_dataset(h5_path)


# Found 101 trajectories in train.h5

# ⇢ Keys under trajectory '0': ['cells', 'mesh_pos', 'node_type', 'world_pos']

#      cells: shape (400, 2564, 4), dtype int32
#  node_type: shape (400, 840, 1), dtype int32
#   ↳ unique node_type values (t=0): [0 1 3]
#   mesh_pos: shape (400, 840, 3), dtype float32
#  world_pos: shape (400, 840, 3), dtype float32

# Done.