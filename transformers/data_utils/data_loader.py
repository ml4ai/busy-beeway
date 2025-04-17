from collections import defaultdict
from typing import (
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import torch
from torch.utils.data import Subset

import h5py


class Pref_H5Dataset(torch.utils.data.Dataset):
    # combined = true means this is a virtual dataset of combined data files
    # batch size is set my target participant data
    # Randomly matches indexes of mixed participants to target participant
    def __init__(
        self,
        target_p_file,
        mixed_p_file,
        m_idxs,
        max_episode_length,
    ):
        super(Pref_H5Dataset, self).__init__()
        self.target_p_file = target_p_file
        self.mixed_p_file = mixed_p_file
        self.m_idxs = m_idxs
        self._max_episode_length = max_episode_length
        with h5py.File(self.target_p_file, "r") as f:
            with h5py.File(self.mixed_p_file, "r") as g:
                self._sts_shape = g["states"].shape
                self._acts_shape = g["actions"].shape

    # Target data is denoted with a 2, since the labels are all 1 for the target
    def open_hdf5(self):
        self.h5_mixed_file = h5py.File(self.mixed_p_file, "r")
        self.states = self.h5_mixed_file["states"]
        self.actions = self.h5_mixed_file["actions"]
        self.timesteps = self.h5_mixed_file["timesteps"]
        self.attn_mask = self.h5_mixed_file["attn_mask"]

        self.h5_target_file = h5py.File(self.target_p_file, "r")
        self.states_2 = self.h5_target_file["states"]
        self.actions_2 = self.h5_target_file["actions"]
        self.timesteps_2 = self.h5_target_file["timesteps"]
        self.attn_mask_2 = self.h5_target_file["attn_mask"]
        self.labels = self.h5_target_file["labels"]

    def __getitem__(self, index):
        if not hasattr(self, "h5_mixed_file"):
            self.open_hdf5()
        return (
            self.states[index, ...],
            self.actions[index, ...],
            self.timesteps[index, ...],
            self.attn_mask[index, ...],
            self.states_2[self.m_idxs[index], ...],
            self.actions_2[self.m_idxs[index], ...],
            self.timesteps_2[self.m_idxs[index], ...],
            self.attn_mask_2[self.m_idxs[index], ...],
            self.labels[self.m_idxs[index]],
        )

    def __len__(self):
        return self._sts_shape[0]

    def shapes(self):
        return self._sts_shape, self._acts_shape

    def max_episode_length(self):
        return self._max_episode_length


class Pref_H5Dataset_minari(torch.utils.data.Dataset):
    # combined = true means this is a virtual dataset of combined data files
    # batch size is set my target participant data
    # Randomly matches indexes of mixed participants to target participant
    def __init__(self, datafile, max_episode_length=None):
        super(Pref_H5Dataset_minari, self).__init__()
        self.datafile = datafile
        with h5py.File(self.datafile, "r") as f:
            if max_episode_length is None:
                self._max_episode_length = np.max(
                    [np.max(f["timesteps"][:]), np.max(f["timesteps_2"][:])]
                )
            else:
                self._max_episode_length = max_episode_length

            self._sts_shape = f["states"].shape
            self._acts_shape = f["actions"].shape

    # Target data is denoted with a 2, since the labels are all 1 for the target
    def open_hdf5(self):
        self.h5_file = h5py.File(self.datafile, "r")
        self.states = self.h5_file["states"]
        self.actions = self.h5_file["actions"]
        self.timesteps = self.h5_file["timesteps"]
        self.attn_mask = self.h5_file["attn_mask"]

        self.states_2 = self.h5_file["states_2"]
        self.actions_2 = self.h5_file["actions_2"]
        self.timesteps_2 = self.h5_file["timesteps_2"]
        self.attn_mask_2 = self.h5_file["attn_mask_2"]
        self.labels = self.h5_file["labels"]

    def __getitem__(self, index):
        if not hasattr(self, "h5_file"):
            self.open_hdf5()
        return (
            self.states[index, ...],
            self.actions[index, ...],
            self.timesteps[index, ...],
            self.attn_mask[index, ...],
            self.states_2[index, ...],
            self.actions_2[index, ...],
            self.timesteps_2[index, ...],
            self.attn_mask_2[index, ...],
            self.labels[index],
        )

    def __len__(self):
        return self._sts_shape[0]

    def shapes(self):
        return self._sts_shape, self._acts_shape

    def max_episode_length(self):
        return self._max_episode_length


class Dec_H5Dataset(torch.utils.data.Dataset):
    # combined = true means this is a virtual dataset of combined data files
    # the data tag is used for return_to_go if there are multiple in the file.
    # The task returns flag overwrite normalized_returns flag
    def __init__(self, file_path, normalized_returns=True, task_returns=False):
        super(Dec_H5Dataset, self).__init__()
        self.file_path = file_path
        self.normalized_returns = normalized_returns
        self.task_returns = task_returns
        with h5py.File(self.file_path, "r") as f:
            self._sts_shape = f["states"].shape
            self._acts_shape = f["actions"].shape
            self._max_episode_length = np.max(f["timesteps"][:])

    def open_hdf5(self):
        self.h5_file = h5py.File(self.file_path, "r")
        self.states = self.h5_file["states"]
        self.actions = self.h5_file["actions"]
        self.timesteps = self.h5_file["timesteps"]
        self.attn_mask = self.h5_file["attn_mask"]
        if self.task_returns:
            self.returns = self.h5_file["task_returns"]
        else:
            if self.normalized_returns:
                self.returns = self.h5_file["n_returns"]
            else:
                self.returns = self.h5_file["raw_returns"]

    def __getitem__(self, index):
        if not hasattr(self, "h5_file"):
            self.open_hdf5()
        return (
            self.states[index, ...],
            self.actions[index, ...],
            self.timesteps[index, ...],
            self.attn_mask[index, ...],
            self.returns[index, ...],
        )

    def __len__(self):
        return self._sts_shape[0]

    def shapes(self):
        return self._sts_shape, self._acts_shape

    def max_episode_length(self):
        return self._max_episode_length


class IQL_H5Dataset(torch.utils.data.Dataset):
    # The task rewards flag overwrite normalized_rewards flag
    # Some environment developers recommended adjusting the task reward function by some constant,
    # this can be set through reward_adjustment.
    def __init__(
        self,
        file_path,
        normalized_rewards=True,
        task_rewards=False,
        reward_adjustment=0.0,
    ):
        super(IQL_H5Dataset, self).__init__()
        self.file_path = file_path
        self.normalized_rewards = normalized_rewards
        self.task_rewards = task_rewards
        self.reward_adjustment = reward_adjustment
        with h5py.File(self.file_path, "r") as f:
            self._sts_shape = f["states"].shape
            self._acts_shape = f["actions"].shape

    def open_hdf5(self):
        self.h5_file = h5py.File(self.file_path, "r")
        self.states = self.h5_file["states"]
        self.next_states = self.h5_file["next_states"]
        self.actions = self.h5_file["actions"]
        self.attn_mask = self.h5_file["attn_mask"]
        if self.task_rewards:
            self.rewards = self.h5_file["task_rewards"]
        else:
            if self.normalized_rewards:
                self.rewards = self.h5_file["n_rewards"]
            else:
                self.rewards = self.h5_file["rewards"]

    def __getitem__(self, index):
        if not hasattr(self, "h5_file"):
            self.open_hdf5()
        return (
            self.states[index, ...],
            self.next_states[index, ...],
            self.actions[index, ...],
            self.attn_mask[index, ...],
            self.rewards[index, ...] + self.reward_adjustment,
        )

    def __len__(self):
        return self._sts_shape[0]

    def shapes(self):
        return self._sts_shape, self._acts_shape
