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
        description="Combines preference data for multiple participants. \nThe combined datasets are virtual \n(see https://docs.h5py.org/en/latest/vds.html) \nand are externally linked to the \ninvidiual preference data files. \nRenaming, Modifying, deleting, or moving any of the \nfiles that comprise the virtual datasets \nmay cause undefined behavior.",
        formatter_class=StructuredFormatter,
    )
    parser.add_argument(
        "p_id",
        metavar="PID",
        type=str,
        help="A .txt file containing a list of Participant IDs to process",
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        default="~/busy-beeway/transformers/preference_data_1",
        help="Data Directory for participant preference data.",
    )
    parser.add_argument(
        "-s",
        "--save_file",
        type=str,
        default="~/busy-beeway/transformers/preference_data/bbway1.hdf5",
        help="Name and location of outputted file",
    )
    args = parser.parse_args(argv)
    p_id = load_list(args.p_id)
    data_dir = os.path.expanduser(args.data_dir)
    save_file = os.path.expanduser(args.save_file)
    data_sizes = []
    segment_sizes = []
    feature_sizes = []
    for p in p_id:
        with h5py.File(f"{data_dir}/{p}.hdf5") as f:
            d, s, f = f["observations"].shape
            data_sizes.append(d)
            segment_sizes.append(s)
            feature_sizes.append(f)
    if not np.all(np.array(segment_sizes) == segment_sizes[0]):
        raise ValueError("All segment lengths must be the same!")
    if not np.all(np.array(feature_sizes) == feature_sizes[0]):
        raise ValueError("The feature dimension must be the same across all datasets!")

    d_s_sum = sum(data_sizes)
    o_layout = h5py.VirtualLayout(
        shape=(d_s_sum, segment_sizes[0], feature_sizes[0]), dtype="<f8"
    )
    t_layout = h5py.VirtualLayout(shape=(d_s_sum, segment_sizes[0]), dtype="<i4")
    am_layout = h5py.VirtualLayout(shape=(d_s_sum, segment_sizes[0]), dtype="<f4")

    o_2_layout = h5py.VirtualLayout(
        shape=(d_s_sum, segment_sizes[0], feature_sizes[0]), dtype="<f8"
    )
    t_2_layout = h5py.VirtualLayout(shape=(d_s_sum, segment_sizes[0]), dtype="<i4")
    am_2_layout = h5py.VirtualLayout(shape=(d_s_sum, segment_sizes[0]), dtype="<f4")

    l_layout = h5py.VirtualLayout(shape=(d_s_sum,), dtype="<f8")
    prev_size = 0
    for i, p in enumerate(p_id):
        o_vsource = h5py.VirtualSource(
            f"{data_dir}/{p}.hdf5",
            "observations",
            shape=(data_sizes[i], segment_sizes[i], feature_sizes[i]),
        )
        t_vsource = h5py.VirtualSource(
            f"{data_dir}/{p}.hdf5", "timesteps", shape=(data_sizes[i], segment_sizes[i])
        )
        am_vsource = h5py.VirtualSource(
            f"{data_dir}/{p}.hdf5", "attn_mask", shape=(data_sizes[i], segment_sizes[i])
        )

        o_2_vsource = h5py.VirtualSource(
            f"{data_dir}/{p}.hdf5",
            "observations_2",
            shape=(data_sizes[i], segment_sizes[i], feature_sizes[i]),
        )
        t_2_vsource = h5py.VirtualSource(
            f"{data_dir}/{p}.hdf5",
            "timesteps_2",
            shape=(data_sizes[i], segment_sizes[i]),
        )
        am_2_vsource = h5py.VirtualSource(
            f"{data_dir}/{p}.hdf5",
            "attn_mask_2",
            shape=(data_sizes[i], segment_sizes[i]),
        )

        l_vsource = h5py.VirtualSource(
            f"{data_dir}/{p}.hdf5", "labels", shape=(data_sizes[i],)
        )

        o_layout[prev_size : (prev_size + data_sizes[i]), :, :] = o_vsource
        t_layout[prev_size : (prev_size + data_sizes[i]), :] = t_vsource
        am_layout[prev_size : (prev_size + data_sizes[i]), :] = am_vsource

        o_2_layout[prev_size : (prev_size + data_sizes[i]), :, :] = o_2_vsource
        t_2_layout[prev_size : (prev_size + data_sizes[i]), :] = t_2_vsource
        am_2_layout[prev_size : (prev_size + data_sizes[i]), :] = am_2_vsource

        l_layout[prev_size : (prev_size + data_sizes[i])] = l_vsource

        prev_size += data_sizes[i]

    with h5py.File(save_file, "w") as f:
        f.create_virtual_dataset("observations", o_layout)
        f.create_virtual_dataset("timesteps", t_layout)
        f.create_virtual_dataset("attn_mask", am_layout)

        f.create_virtual_dataset("observations_2", o_2_layout)
        f.create_virtual_dataset("timesteps_2", t_2_layout)
        f.create_virtual_dataset("attn_mask_2", am_2_layout)
        
        f.create_virtual_dataset("labels", l_layout)
    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
