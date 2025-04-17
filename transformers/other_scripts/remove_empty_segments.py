import argparse
import os
import sys

import h5py
import jax.numpy as jnp
import jax
from tqdm import tqdm

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.abspath("../.."))

from argformat import StructuredFormatter

from transformers.data_utils.bb_data_loading import load_list


def main(argv):
    parser = argparse.ArgumentParser(
        description="Removes empty trajectory segments (am == 0 for all timesteps) from the data",
        formatter_class=StructuredFormatter,
    )
    parser.add_argument(
        "participants",
        metavar="P",
        type=str,
        help="Participant ID or a txt file listing Participant IDs",
    )
    parser.add_argument(
        "data",
        metavar="D",
        type=str,
        help="Directory with data",
    )
    args = parser.parse_args(argv)
    data = os.path.expanduser(args.data)
    p_id = os.path.expanduser(args.participants)
    if p_id.endswith(".txt"):
        P = load_list(p_id)
        for p_id in tqdm(P):
            with h5py.File(f"{data}/{p_id}.hdf5", "a") as f:
                sts = f["states"][:]
                acts = f["actions"][:]
                ts = f["timesteps"][:]
                am = f["attn_mask"][:]
                del_idx = jnp.argwhere(jnp.all(am == 0,axis=1))[:, 0]
                sts = jnp.delete(
                    sts, del_idx, axis=0
                )
                acts = jnp.delete(
                    acts, del_idx, axis=0
                )
                ts = jnp.delete(
                    ts, del_idx, axis=0
                )
                am = jnp.delete(
                    am, del_idx, axis=0
                )
                
                del f["states"]
                f.create_dataset("states", data=sts, chunks=True)

                del f["actions"]
                f.create_dataset("actions", data=acts, chunks=True)

                del f["timesteps"]
                f.create_dataset("timesteps", data=ts, chunks=True)

                del f["attn_mask"]
                f.create_dataset("attn_mask", data=am, chunks=True)

    else:
        with h5py.File(f"{data}/{p_id}.hdf5", "a") as f:
            sts = f["states"][:]
            acts = f["actions"][:]
            ts = f["timesteps"][:]
            am = f["attn_mask"][:]

            del_idx = jnp.argwhere(jnp.all(am == 0,axis=1))[:, 0]
            sts = jnp.delete(
                sts, del_idx, axis=0
            )
            acts = jnp.delete(
                acts, del_idx, axis=0
            )
            ts = jnp.delete(
                ts, del_idx, axis=0
            )
            am = jnp.delete(
                am, del_idx, axis=0
            )
            
            del f["states"]
            f.create_dataset("states", data=sts, chunks=True)

            del f["actions"]
            f.create_dataset("actions", data=acts, chunks=True)

            del f["timesteps"]
            f.create_dataset("timesteps", data=ts, chunks=True)

            del f["attn_mask"]
            f.create_dataset("attn_mask", data=am, chunks=True)

    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
