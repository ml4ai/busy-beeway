import h5py
import numpy as np
import torch


class Pref_H5Dataset(torch.utils.data.Dataset):

    def __init__(self, file_path):
        super(Pref_H5Dataset, self).__init__()
        self.file_path = file_path
        with h5py.File(self.file_path, "r") as f:
            self._shape = f["observations"].shape
            self._max_episode_length = np.max(f["timesteps"][:])

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
            self.observations[index, ...],
            self.timesteps[index, ...],
            self.attn_mask[index, ...],
            self.observations_2[index, ...],
            self.timesteps_2[index, ...],
            self.attn_mask_2[index, ...],
            self.labels[index],
        )

    def __len__(self):
        return self._shape[0]

    def obs_shape(self):
        return self._shape

    def max_episode_length(self):
        return self._max_episode_length
