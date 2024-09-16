import h5py
import torch
import numpy as np


class Pref_H5Dataset(torch.utils.data.Dataset):

    def __init__(self, file_path):
        super(Pref_H5Dataset, self).__init__()
        h5_file = h5py.File(file_path, "r")
        self.observations = h5_file["observations"]
        self.timesteps = h5_file["timesteps"]
        self.attn_mask = h5_file["attn_mask"]
        self.observations_2 = h5_file["observations_2"]
        self.timesteps_2 = h5_file["timesteps_2"]
        self.attn_mask_2 = h5_file["attn_mask_2"]
        self.labels = h5_file["labels"]

    def __getitem__(self, index):

        return (
            torch.from_numpy(self.observations[index, ...]).float(),
            torch.from_numpy(self.timesteps[index, ...]).float(),
            torch.from_numpy(self.attn_mask[index, ...]).float(),
            torch.from_numpy(self.observations_2[index, ...]).float(),
            torch.from_numpy(self.timesteps_2[index, ...]).float(),
            torch.from_numpy(self.attn_mask_2[index, ...]).float(),
            torch.from_numpy(np.array(self.labels[index])).int(),
        )

    def __len__(self):
        return self.observations.shape[0]

    def obs_shape(self):
        return self.observations.shape
