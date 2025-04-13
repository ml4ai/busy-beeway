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

import h5py
import numpy as np
import torch
from torch.utils.data import Subset

class Pref_H5Dataset(torch.utils.data.Dataset):
    # combined = true means this is a virtual dataset of combined data files
    # batch size is set my target participant data
    # Randomly matches indexes of mixed participants to target participant
    def __init__(
        self,
        target_p_file,
        mixed_p_file,
        rng=np.random.default_rng(),
        max_episode_length=None,
    ):
        super(Pref_H5Dataset, self).__init__()
        self.target_p_file = target_p_file
        self.mixed_p_file = mixed_p_file
        with h5py.File(self.target_p_file, "r") as f:
            if max_episode_length is None:
                self._max_episode_length = np.max(f["timesteps"][:])
            else:
                self._max_episode_length = max_episode_length
            # if combined:
            #     self._c_idx = {}
            #     for key, val in f.attrs.items():
            #         self._c_idx[key] = val
            #     self._c_n = len(self._c_idx)
            # else:
            #     self._c_idx = None
            #     self._c_n = 1

            with h5py.File(self.mixed_p_file, "r") as g:
                self._sts_shape = g["states"].shape
                self._acts_shape = g["actions"].shape
                m_size = g["states"].shape[0]
                # self.labels = []
                self.m_idxs = np.zeros(m_size)
                for m in range(m_size):
                    if max_episode_length is None:
                        self._max_episode_length = max(
                            np.max(g["timesteps"][m, :]), self._max_episode_length
                        )

                    m_static = g["states"][m, 0, -4:]
                    t_static = f["states"][:, 0, -4:]
                    matches = np.argwhere(np.all(t_static == m_static, axis=1))[:, 0]
                    if matches.shape[0] > 0:
                        self.m_idxs[m] = rng.choice(matches)
                    else:
                        self.m_idxs[m] = rng.choice(t_static.shape[0])
                #     if np.all(f["actions"][self.m_idxs[m], :, 2] == 0) and np.all(
                #         g["actions"][m, :, 2] == 0
                #     ):
                #         self.labels.append(0.5)
                #     else:
                #         self.labels.append(1.0)
                # self.labels = np.asarray(self.labels)
                # if combined:
                #     self._c_idx = {}
                #     for key, val in f.attrs.items():
                #         self._c_idx[key] = val
                #     self._c_n = len(self._c_idx)
                # else:
                #     self._c_idx = None
                #     self._c_n = 1

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

    # # None if not a combined dataset
    # def c_idx(self):
    #     return self._c_idx

    # # 1 if not a combined dataset
    # def c_num(self):
    #     return self._c_n


class Pref_H5Dataset_minari(torch.utils.data.Dataset):
    # combined = true means this is a virtual dataset of combined data files
    # batch size is set my target participant data
    # Randomly matches indexes of mixed participants to target participant
    def __init__(self, datafile, max_episode_length=None):
        super(Pref_H5Dataset_minari, self).__init__()
        self.datafile = datafile
        with h5py.File(self.datafile, "r") as f:
            if max_episode_length is None:
                self._max_episode_length = max(
                    np.max(f["timesteps"][:]), np.max(f["timesteps_2"][:])
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
        self._max_episode_length = 0
        with h5py.File(self.file_path, "r") as f:
            self._sts_shape = f["states"].shape
            self._acts_shape = f["actions"].shape
            size = f["states"].shape[0]
            for i in range(size):
                self._max_episode_length = max(
                    np.max(f["timesteps"][i, ...]), self._max_episode_length
                )

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


