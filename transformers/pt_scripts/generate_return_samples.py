import argparse
import os
import sys

import h5py
import jax
import jax.numpy as jnp

jax.config.update("jax_platforms", "cpu")

sys.path.insert(0, os.path.abspath("../.."))

from argformat import StructuredFormatter

from transformers.training.utils import load_pickle


def main(argv):
    parser = argparse.ArgumentParser(
        description="Generates Monte Carlo Targets for Value Function Approximations \nusing precollected trajectory samples. \nThis physically resaves the data to the given data file.",
        formatter_class=StructuredFormatter,
    )
    parser.add_argument(
        "reward",
        metavar="R",
        type=str,
        help="File with Reward function (as pickled dictionary)",
    )
    parser.add_argument(
        "data",
        metavar="D",
        type=str,
        help="File with sample Trajectories",
    )
    parser.add_argument(
        "-s",
        "--set_2",
        action="store_true",
        help="This uses the real data samples \nversus the random policy samples.",
    )
    parser.add_argument(
        "-t",
        "--data_tag",
        type=str,
        default=None,
        help="adds identifier to return_to_go dataset",
    )
    args = parser.parse_args(argv)
    reward = os.path.expanduser(args.reward)
    data = os.path.expanduser(args.data)
    if args.data_tag is not None:
        r_l = f"return_to_go_{args.data_tag}"
    else:
        r_l = "return_to_go"
    r_model = load_pickle(reward)["model"]
    with h5py.File(data, "r+") as f:
        if args.set_2:
            sts = f["states_2"][:]
            acts = f["actions_2"][:]
            ts = f["timesteps_2"][:]
            am = f["attn_mask_2"][:]
            r_l = f"{r_l}_2"
        else:
            sts = f["states"][:]
            acts = f["actions"][:]
            ts = f["timesteps"][:]
            am = f["attn_mask"][:]
        del f["states_2"]
        del f["actions_2"]
        del f["timesteps_2"]
        del f["attn_mask_2"]
        del f["states"]
        del f["actions"]
        del f["timesteps"]
        del f["attn_mask"]
        seq_length = sts.shape[1]
        return_to_go = []
        for i in range(seq_length):
            preds, _ = r_model._train_state.apply_fn(
                r_model._train_state.params,
                sts[:, : (i + 1), :],
                acts[:, : (i + 1), :],
                ts[:, : (i + 1)],
                training=False,
                attn_mask=am[:, : (i + 1)],
            )
            if return_to_go:
                return_to_go.append(return_to_go[-1] + preds["value"][:, 0, -1])
            else:
                return_to_go.append(preds["value"][:, 0, -1])
        return_to_go = jnp.concatenate(return_to_go, axis=1)
        if jnp.any(jnp.isnan(return_to_go)):
            sts = np.delete(sts,jnp.unique(jnp.argwhere(jnp.isnan(return_to_go))[:,0]),axis=0)
            acts = np.delete(acts,jnp.unique(jnp.argwhere(jnp.isnan(return_to_go))[:,0]),axis=0)
            ts = np.delete(ts,jnp.unique(jnp.argwhere(jnp.isnan(return_to_go))[:,0]),axis=0)
            am = np.delete(am,jnp.unique(jnp.argwhere(jnp.isnan(return_to_go))[:,0]),axis=0)
            return_to_go = np.delete(return_to_go,jnp.unique(jnp.argwhere(jnp.isnan(return_to_go))[:,0]),axis=0)
        f.create_dataset("states",data=sts,chunks=True)
        f.create_dataset("actions",data=acts,chunks=True)
        f.create_dataset("timesteps",data=ts,chunks=True)
        f.create_dataset("attn_mask",data=ts,chunks=True)
        f.create_dataset(r_l, data=return_to_go,chunks=True)
    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
