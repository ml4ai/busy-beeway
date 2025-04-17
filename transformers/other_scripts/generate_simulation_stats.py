import argparse
import os
import sys

import h5py
import jax
import jax.numpy as jnp
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx

jax.config.update("jax_platforms", "cpu")

sys.path.insert(0, os.path.abspath("../.."))

from argformat import StructuredFormatter

from transformers.evaluation.eval_episodes import bb_record_episode
from transformers.replayer.replayer import animate_run, load_stats
from transformers.models.pref_transformer import load_PT
from transformers.models.dec_transformer import load_DT
from transformers.data_utils.bb_data_loading import load_participant_data_p


def main(argv):
    parser = argparse.ArgumentParser(
        description="Generates comparisons for real versus generated behavior",
        formatter_class=StructuredFormatter,
    )
    parser.add_argument(
        "dt",
        metavar="D",
        type=str,
        help="File with Decision Transformer model",
    )
    parser.add_argument(
        "pt",
        metavar="P",
        type=str,
        help="File with Preference Transformer model",
    )
    parser.add_argument(
        "stats",
        metavar="S",
        type=str,
        help="File with Obstacle Stats",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=25102,
        help="Random seed",
    )
    parser.add_argument(
        "-e",
        "--episodes",
        type=int,
        default=1000,
        help="Number of Episodes",
    )
    parser.add_argument(
        "-t",
        "--target_return",
        type=float,
        default=8000.0,
        help="Target return",
    )
    parser.add_argument(
        "-m",
        "--horizon",
        type=int,
        default=400,
        help="Horizon",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="~/busy-beeway/transformers/sim_stats.hdf5",
        help="Output file",
    )
    args = parser.parse_args(argv)
    dt = os.path.expanduser(args.dt)
    pt = os.path.expanduser(args.pt)
    stats = os.path.expanduser(args.stats)
    seed = args.seed
    episodes = args.episodes
    target_return = args.target_return
    horizon = args.horizon
    output = os.path.expanduser(args.output)
    checkpointer = ocp.Checkpointer(ocp.CompositeCheckpointHandler())
    d_model = load_DT(os.path.expanduser(dt), checkpointer, on_cpu=True)
    r_model = load_PT(os.path.expanduser(pt), checkpointer, on_cpu=True)
    r_model = nnx.jit(r_model, static_argnums=4)
    checkpointer.close()
    move_stats = load_stats(stats)
    rng = np.random.default_rng(seed)
    successes = 0.0
    end_goal_dist = []
    frames = []
    returns = []
    for i in tqdm(range(episodes)):
        e_r, e_l, d = bb_record_episode(
            d_model, r_model, move_stats, 100, target_return, horizon, rng=rng
        )
        if d["reached_goal"]:
            successes += 1.0
        posX = d["player"]["posX"].to_numpy()[-1]
        posY = d["player"]["posY"].to_numpy()[-1]
        goal = d["goal"]
        end_goal_dist.append(np.sqrt(((goal[0] - posX) ** 2) + ((goal[1] - posY) ** 2)))
        frames.append(e_l + 1)
        returns.append(e_r[0][0])
        del d
        jax.clear_caches()

    with h5py.File(output, "a") as f:
        if "successes" in f:
            f["successes"][()] = f["successes"][()] + successes
        else:
            f.create_dataset("successes", data=successes)

        if "end_goal_distance" in f:
            old_end_goal_dist = f["end_goal_distance"][:]
            del f["end_goal_distance"]
            f.create_dataset(
                "end_goal_distance",
                data=np.concatenate([old_end_goal_dist, end_goal_dist]),
            )
        else:
            f.create_dataset("end_goal_distance", data=np.asarray(end_goal_dist))

        if "frames" in f:
            old_frames = f["frames"][:]
            del f["frames"]
            f.create_dataset("frames", data=np.concatenate([old_frames, frames]))
        else:
            f.create_dataset("frames", data=np.asarray(frames))

        if "returns" in f:
            old_returns = f["returns"][:]
            del f["returns"]
            f.create_dataset("returns", data=np.concatenate([old_returns, returns]))
        else:
            f.create_dataset("returns", data=np.asarray(returns))

    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
