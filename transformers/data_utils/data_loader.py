import h5py
import numpy as np
import torch


class Pref_H5Dataset(torch.utils.data.Dataset):

    def __init__(self, file_path):
        super(Pref_H5Dataset, self).__init__()
        self.file_path = file_path
        with h5py.File(self.file_path, "r") as f:
            self.shape = f["observations"].shape

    def open_hdf5(self):
        self.h5_file = h5py.File(self.file_path, "r")
        self.observations = self.h5_file["observations"]
        self.timesteps = self.h5_file["timesteps"]
        self.attn_mask = self.h5_file["attn_mask"]
        self.observations_2 = self.h5_file["observations_2"]
        self.timesteps_2 = self.h5_file["timesteps_2"]
        self.attn_mask_2 = self.h5_file["attn_mask_2"]
        self.labels = self.h5_file["labels"]

    def __getitem__(self, index):
        if not hasattr(self, "h5_file"):
            self.open_hdf5()
        return (
            np.asarray(self.observations[index, ...], dtype=np.float32),
            np.asarray(self.timesteps[index, ...], dtype=np.float32),
            np.asarray(self.attn_mask[index, ...], dtype=np.float32),
            np.asarray(self.observations_2[index, ...], dtype=np.float32),
            np.asarray(self.timesteps_2[index, ...], dtype=np.float32),
            np.asarray(self.attn_mask_2[index, ...], dtype=np.float32),
            np.asarray(self.labels[index], dtype=np.float32),
        )

    def __len__(self):
        return self.shape[0]

    def obs_shape(self):
        return self.shape
