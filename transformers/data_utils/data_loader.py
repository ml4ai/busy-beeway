from collections import defaultdict
import itertools
import math
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
from torch import default_generator, Generator, randperm, Tensor
from torch.utils.data import DataLoader, Sampler, BatchSampler
import h5py


class RandomBatchSampler(Sampler):
    """Sampling class to create random sequential batches from a given dataset
    E.g. if data is [1,2,3,4] with bs=2. Then first batch, [[1,2], [3,4]] then shuffle batches -> [[3,4],[1,2]]
    This is useful for cases when you are interested in 'weak shuffling'
    :param dataset: dataset you want to batch
    :type dataset: torch.utils.data.Dataset
    :param batch_size: batch size
    :type batch_size: int
    :returns: generator object of shuffled batch indices
    """

    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.dataset_length = len(dataset)
        self.n_batches = self.dataset_length / self.batch_size
        self.batch_ids = torch.randperm(int(self.n_batches))

    def __len__(self):
        return self.batch_size

    def __iter__(self):
        for id in self.batch_ids:
            idx = torch.arange(id * self.batch_size, (id + 1) * self.batch_size)
            for index in idx:
                yield int(index)
        if int(self.n_batches) < self.n_batches:
            idx = torch.arange(
                int(self.n_batches) * self.batch_size, self.dataset_length
            )
            for index in idx:
                yield int(index)


# DOES WEAK SHUFFLING! CAN INTRODUCE BIAS IF NOT |data| >> batch_size
def fast_loader(dataset, batch_size=32, drop_last=False, transforms=None):
    """Implements fast loading by taking advantage of .h5 dataset
    The .h5 dataset has a speed bottleneck that scales (roughly) linearly with the number
    of calls made to it. This is because when queries are made to it, a search is made to find
    the data item at that index. However, once the start index has been found, taking the next items
    does not require any more significant computation. So indexing data[start_index: start_index+batch_size]
    is almost the same as just data[start_index]. The fast loading scheme takes advantage of this. However,
    because the goal is NOT to load the entirety of the data in memory at once, weak shuffling is used instead of
    strong shuffling.
    :param dataset: a dataset that loads data from .h5 files
    :type dataset: torch.utils.data.Dataset
    :param batch_size: size of data to batch
    :type batch_size: int
    :param drop_last: flag to indicate if last batch will be dropped (if size < batch_size)
    :type drop_last: bool
    :returns: dataloading that queries from data using shuffled batches
    :rtype: torch.utils.data.DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=None,  # must be disabled when using samplers
        sampler=BatchSampler(
            RandomBatchSampler(dataset, batch_size),
            batch_size=batch_size,
            drop_last=drop_last,
        ),
    )


class Pref_H5Dataset(torch.utils.data.Dataset):
    def __init__(self, datafile, max_episode_length=None):
        super(Pref_H5Dataset, self).__init__()
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


def sorted_random_split(
    dataset: Dataset[_T],
    lengths: Sequence[Union[int, float]],
    generator: Optional[Generator] = default_generator,
) -> List[Subset[_T]]:
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.

    Optionally fix the generator for reproducible results, e.g.:

    Example:
        >>> # xdoctest: +SKIP
        >>> generator1 = torch.Generator().manual_seed(42)
        >>> generator2 = torch.Generator().manual_seed(42)
        >>> random_split(range(10), [3, 7], generator=generator1)
        >>> random_split(range(30), [0.3, 0.3, 0.4], generator=generator2)

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: List[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(
                    f"Length of split at index {i} is 0. "
                    f"This might result in an empty dataset."
                )

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):  # type: ignore[arg-type]
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )

    indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[arg-type, call-overload]
    lengths = cast(Sequence[int], lengths)
    return [
        Subset(dataset, torch.sort(indices[offset - length : offset]))
        for offset, length in zip(itertools.accumulate(lengths), lengths)
    ]
