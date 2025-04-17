import argparse
import sys
import os

sys.path.insert(0, os.path.abspath("../.."))
from pathlib import Path

import h5py
from argformat import StructuredFormatter
import numpy as np
from transformers.data_utils.bb_data_loading import load_list


def main(argv):
    parser = argparse.ArgumentParser(
        description="Combines state data for files. \nThe combined datasets are virtual \n(see https://docs.h5py.org/en/latest/vds.html) \nand are externally linked to the \ninvidiual preference data files. \nRenaming, Modifying, deleting, or moving any of the \nfiles that comprise the virtual datasets \nmay cause undefined behavior.",
        formatter_class=StructuredFormatter,
    )
    parser.add_argument(
        "p_id",
        metavar="PID",
        type=str,
        help="A .txt file containing a list of filenames (no extensions) to process",
    )
    parser.add_argument(
        "-e",
        "--exclude",
        type=str,
        default=None,
        help="Exclude participant (or filename) from p_id",
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        default="~/busy-beeway/transformers/state_data_1",
        help="Data Directory for participant state data.",
    )
    parser.add_argument(
        "-s",
        "--save_file",
        type=str,
        default="~/busy-beeway/transformers/state_data_1/bbway1.hdf5",
        help="Name and location of outputted file",
    )
    parser.add_argument(
        "-n",
        "--normalized_rewards",
        action="store_true",
        help="Combine Normalized Rewards.",
    )
    args = parser.parse_args(argv)
    p_id = load_list(args.p_id)
    data_dir = os.path.expanduser(args.data_dir)
    save_file = os.path.expanduser(args.save_file)
    if args.exclude is not None:
        p_id.remove(args.exclude)
    data_sizes = []
    state_sizes = []
    action_sizes = []
    for p in p_id:
        with h5py.File(f"{data_dir}/{p}.hdf5") as f:
            d, st = f["states"].shape
            act = f["actions"].shape[1]
            data_sizes.append(d)
            state_sizes.append(st)
            action_sizes.append(act)
    if not np.all(np.array(state_sizes) == state_sizes[0]):
        raise ValueError("The state dimension must be the same across all datasets!")
    if not np.all(np.array(action_sizes) == action_sizes[0]):
        raise ValueError("The state dimension must be the same across all datasets!")

    d_s_sum = sum(data_sizes)
    st_layout = h5py.VirtualLayout(shape=(d_s_sum, state_sizes[0]), dtype="<f8")

    next_st_layout = h5py.VirtualLayout(shape=(d_s_sum, state_sizes[0]), dtype="<f8")

    act_layout = h5py.VirtualLayout(shape=(d_s_sum, action_sizes[0]), dtype="<f8")

    am_layout = h5py.VirtualLayout(shape=(d_s_sum,), dtype="<f4")

    r_layout = h5py.VirtualLayout(shape=(d_s_sum,), dtype="<f8")

    if args.normalized_rewards:
        n_r_layout = h5py.VirtualLayout(shape=(d_s_sum,), dtype="<f8")

    prev_size = 0
    p_idx = []
    for i, p in enumerate(p_id):
        st_vsource = h5py.VirtualSource(
            f"{data_dir}/{p}.hdf5",
            "states",
            shape=(data_sizes[i], state_sizes[i]),
        )

        next_st_vsource = h5py.VirtualSource(
            f"{data_dir}/{p}.hdf5",
            "next_states",
            shape=(data_sizes[i], state_sizes[i]),
        )

        act_vsource = h5py.VirtualSource(
            f"{data_dir}/{p}.hdf5",
            "actions",
            shape=(data_sizes[i], action_sizes[i]),
        )

        am_vsource = h5py.VirtualSource(
            f"{data_dir}/{p}.hdf5", "attn_mask", shape=(data_sizes[i],)
        )

        r_vsource = h5py.VirtualSource(
            f"{data_dir}/{p}.hdf5", "rewards", shape=(data_sizes[i],)
        )

        st_layout[prev_size : (prev_size + data_sizes[i]), :] = st_vsource
        next_st_layout[prev_size : (prev_size + data_sizes[i]), :] = next_st_vsource
        act_layout[prev_size : (prev_size + data_sizes[i]), :] = act_vsource
        am_layout[prev_size : (prev_size + data_sizes[i])] = am_vsource
        r_layout[prev_size : (prev_size + data_sizes[i])] = r_vsource

        if args.normalized_rewards:
            n_r_vsource = h5py.VirtualSource(
                f"{data_dir}/{p}.hdf5", "n_rewards", shape=(data_sizes[i],)
            )
            n_r_layout[prev_size : (prev_size + data_sizes[i])] = n_r_vsource
        # end of range is exclusive. So 0,100 is really 0 to 99 inclusive.
        p_idx.append(np.array([prev_size, (prev_size + data_sizes[i])]))
        prev_size += data_sizes[i]

    with h5py.File(save_file, "a") as f:
        f.create_virtual_dataset("states", st_layout)
        f.create_virtual_dataset("next_states", next_st_layout)
        f.create_virtual_dataset("actions", act_layout)
        f.create_virtual_dataset("attn_mask", am_layout)
        f.create_virtual_dataset("rewards", r_layout)
        if args.normalized_rewards:
            f.create_virtual_dataset("n_rewards", n_r_layout)
        for i, p in enumerate(p_id):
            f.attrs[p] = p_idx[i]
    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
