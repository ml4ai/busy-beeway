from collections import defaultdict

import h5py
import jax.numpy as jnp
import numpy as np
import torch
from torch.utils.data import Subset


class Pref_H5Dataset(torch.utils.data.Dataset):
    # combined = true means this is a virtual dataset of combined data files
    def __init__(self, file_path, combined=True):
        super(Pref_H5Dataset, self).__init__()
        self.file_path = file_path
        with h5py.File(self.file_path, "r") as f:
            self._sts_shape = f["states"].shape
            self._acts_shape = f["actions"].shape
            self._max_episode_length = np.max(f["timesteps"][:])
            if combined:
                self._c_idx = {}
                for key, val in f.attrs.items():
                    self._c_idx[key] = val
                self._c_n = len(self._c_idx)
            else:
                self._c_idx = None
                self._c_n = 1

    def open_hdf5(self):
        self.h5_file = h5py.File(self.file_path, "r")
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
        return self._shape[0]

    def shapes(self):
        return self._sts_shape, self._acts_shape

    def max_episode_length(self):
        return self._max_episode_length

    # None if not a combined dataset
    def c_idx(self):
        return self._c_idx

    # 1 if not a combined dataset
    def c_num(self):
        return self._c_n


# Returns subset with list of ranges
def create_subset(dataset, n_idx):
    idx = []
    sub_c_idx = {}
    c_idx = dataset.c_idx()
    c_idx_k = list(c_idx.keys())
    prev_r = 0
    for i in n_idx:
        c_range = c_idx[c_idx_k[i]]
        idx.append(range(c_range[0], c_range[1]))
        size = c_range[1] - c_range[0]
        sub_c_idx[c_idx_k[i]] = np.array([prev_r, (prev_r + size)])
        prev_r += size
    return Subset(dataset, np.concatenate(idx)), sub_c_idx


# gen must be a torch generator. If shuffle is false, first training, then validation, then test sets are filled in the
# order that classes appear in the dataset.
def get_train_val_test_split(dataset, train_n, val_n, test_n, shuffle=True, gen=None):
    assert dataset.c_num() == (
        train_n + val_n + test_n
    ), "The training, validation, and test sets must add up to the total amount of classes!"
    if shuffle:
        idx = torch.randperm(dataset.c_num(), generator=gen)
    else:
        idx = np.arange(dataset.c_num())

    tr_idx = idx[0:train_n]
    v_idx = idx[train_n : (train_n + val_n)]
    te_idx = idx[(train_n + val_n) : (train_n + val_n + test_n)]

    return (
        create_subset(dataset, tr_idx),
        create_subset(dataset, v_idx),
        create_subset(dataset, te_idx),
    )


class FewShotBatchSampler(object):

    def __init__(
        self,
        classes,
        N_way,
        K_shot,
        include_query=False,
        shuffle=True,
        shuffle_once=False,
        gen=None,
    ):
        """
        Inputs:
            dataset_targets - PyTorch tensor of the labels of the data elements.
            N_way - Number of classes to sample per batch.
            K_shot - Number of examples to sample per class in the batch.
            include_query - If True, returns batch of size N_way*K_shot*2, which
                            can be split into support and query set. Simplifies
                            the implementation of sampling the same classes but
                            distinct examples for support and query set.
            shuffle - If True, examples and classes are newly shuffled in each
                      iteration (for training)
            shuffle_once - If True, examples and classes are shuffled once in
                           the beginning, but kept constant across iterations
                           (for validation)
        """
        super().__init__()
        self.gen = gen
        self.N_way = N_way
        self.K_shot = K_shot
        self.shuffle = shuffle
        self.include_query = include_query
        if self.include_query:
            self.K_shot *= 2
        self.batch_size = (
            self.N_way * self.K_shot
        )  # Number of overall classes per batch. The last batch might be smaller.

        # Organize examples by class
        self.classes = classes
        self.num_classes = len(self.classes)
        self.indices_per_class = {}
        self.batches_per_class = (
            {}
        )  # Number of K-shot batches that each class can provide
        for k, v in self.classes.items():
            self.indices_per_class[k] = np.arange(v[0], v[1])
            size = self.indices_per_class[k].shape[0]
            self.batches_per_class[k] = int(size // self.K_shot)

        # Create a list of classes from which we select the N classes per batch
        self.i_size = sum(self.batches_per_class.values())
        self.iterations = int(self.i_size // self.N_way)
        self.class_list = [
            c for c in self.classes for _ in range(self.batches_per_class[c])
        ]
        if shuffle_once or self.shuffle:
            self.shuffle_data()
        else:
            # For testing, we iterate over classes instead of shuffling them
            sort_idxs = [
                i + p * self.num_classes
                for i, c in enumerate(self.classes)
                for p in range(self.batches_per_class[c])
            ]
            self.class_list = np.array(self.class_list)[np.argsort(sort_idxs)].tolist()

    def shuffle_data(self):
        # Shuffle the examples per class
        for k in self.classes:
            perm = torch.randperm(
                self.indices_per_class[k].shape[0], generator=self.gen
            )
            self.indices_per_class[k] = self.indices_per_class[k][perm]
        # Shuffle the class list from which we sample. Note that this way of shuffling
        # does not prevent to choose the same class twice in a batch. However, for
        # training and validation, this is not a problem.
        class_list_np = np.array(self.class_list)
        self.class_list = list(
            class_list_np[
                list(torch.randperm(len(self.class_list), generator=self.gen))
            ]
        )

    def __iter__(self):
        # Shuffle data
        if self.shuffle:
            self.shuffle_data()

        # Sample few-shot batches
        start_index = defaultdict(int)
        for it in range(self.iterations):
            class_batch = self.class_list[
                it * self.N_way : (it + 1) * self.N_way
            ]  # Select N class for the batch
            index_batch = []
            for (
                c
            ) in (
                class_batch
            ):  # For each class, select the next K examples and add them to the batch
                index_batch.extend(
                    self.indices_per_class[c][
                        start_index[c] : start_index[c] + self.K_shot
                    ]
                )
                start_index[c] += self.K_shot
            if (
                self.include_query
            ):  # If we return support+query set, sort them so that they are easy to split
                index_batch = index_batch[::2] + index_batch[1::2]
            yield index_batch

    def __len__(self):
        return self.iterations


# This splits class batch into support and query set and reshapes by class. Also converts to jax numpy arrays.
def process_c_batch(batch, N_way, K_shot):
    sts, acts, ts, am, sts2, acts2, ts2, am2, lab = batch

    _, seq_length, st_dim = sts.shape
    act_dim = acts.shape[2]

    s_sts, q_sts = sts.chunk(2, dim=0)
    s_acts, q_acts = acts.chunk(2, dim=0)
    s_ts, q_ts = ts.chunk(2, dim=0)
    s_am, q_am = am.chunk(2, dim=0)
    s_sts2, q_sts2 = sts2.chunk(2, dim=0)
    s_acts2, q_acts2 = acts2.chunk(2, dim=0)
    s_ts2, q_ts2 = ts2.chunk(2, dim=0)
    s_am2, q_am2 = am2.chunk(2, dim=0)
    s_lab, q_lab = lab.chunk(2, dim=0)

    s_sts, q_sts = jnp.asarray(
        s_sts.reshape(N_way, K_shot, seq_length, st_dim)
    ), jnp.asarray(q_sts.reshape(N_way, K_shot, seq_length, st_dim))

    s_acts, q_acts = jnp.asarray(
        s_acts.reshape(N_way, K_shot, seq_length, act_dim)
    ), jnp.asarray(q_acts.reshape(N_way, K_shot, seq_length, act_dim))

    s_ts, q_ts = jnp.asarray(s_ts.reshape(N_way, K_shot, seq_length)), jnp.asarray(
        q_ts.reshape(N_way, K_shot, seq_length)
    )

    s_am, q_am = jnp.asarray(s_am.reshape(N_way, K_shot, seq_length)), jnp.asarray(
        q_am.reshape(N_way, K_shot, seq_length)
    )

    s_sts2, q_sts2 = jnp.asarray(
        s_sts2.reshape(N_way, K_shot, seq_length, st_dim)
    ), jnp.asarray(q_sts2.reshape(N_way, K_shot, seq_length, st_dim))

    s_acts2, q_acts2 = jnp.asarray(
        s_acts2.reshape(N_way, K_shot, seq_length, act_dim)
    ), jnp.asarray(q_acts2.reshape(N_way, K_shot, seq_length, act_dim))

    s_ts2, q_ts2 = jnp.asarray(s_ts2.reshape(N_way, K_shot, seq_length)), jnp.asarray(
        q_ts2.reshape(N_way, K_shot, seq_length)
    )

    s_am2, q_am2 = jnp.asarray(s_am2.reshape(N_way, K_shot, seq_length)), jnp.asarray(
        q_am2.reshape(N_way, K_shot, seq_length)
    )

    s_lab, q_lab = jnp.asarray(s_lab.reshape(N_way, K_shot)), jnp.asarray(
        q_lab.reshape(N_way, K_shot)
    )

    train_batch = (s_sts, s_acts, s_ts, s_am, s_sts2, s_acts2, s_ts2, s_am2, s_lab)
    val_batch = (q_sts, q_acts, q_ts, q_am, q_sts2, q_acts2, q_ts2, q_am2, q_lab)
    return train_batch, val_batch
