import h5py
import numpy as np
import torch


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
            np.asarray(self.observations[index, ...], dtype=np.float32),
            np.asarray(self.timesteps[index, ...],dtype=np.float32),
            np.asarray(self.attn_mask[index, ...],dtype=np.float32),
            np.asarray(self.observations_2[index, ...],dtype=np.float32),
            np.asarray(self.timesteps_2[index, ...],dtype=np.float32),
            np.asarray(self.attn_mask_2[index, ...],dtype=np.float32),
            np.asarray(self.labels[index], dtype=np.float32),
        )

    def __len__(self):
        return self.observations.shape[0]

    def obs_shape(self):
        return self.observations.shape
