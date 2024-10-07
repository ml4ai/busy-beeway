from collections import defaultdict

import h5py
import numpy as np
import jax.numpy as jnp
import torch
from torch.utils.data import Subset


class Pref_H5Dataset(torch.utils.data.Dataset):
    # combined = true means this is a virtual dataset of all participants
    def __init__(self, file_path, combined=True):
        super(Pref_H5Dataset, self).__init__()
        self.file_path = file_path
        with h5py.File(self.file_path, "r") as f:
            self._shape = f["observations"].shape
            self._max_episode_length = np.max(f["timesteps"][:])
            if combined:
                self._p_idx = {}
                for key, val in f.attrs.items():
                    self._p_idx[key] = val
                self._p_n = len(self._p_idx)
            else:
                self._p_idx = None
                self._p_n = 1

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

    # None if not a combined dataset
    def p_idx(self):
        return self._p_idx

    # 1 if not a combined dataset
    def p_num(self):
        return self._p_n


# Returns subset with list of ranges
def create_participant_subset(dataset, n_idx):
    idx = []
    sub_p_idx = {}
    p_idx = dataset.p_idx()
    p_idx_k = list(p_idx.keys())
    prev_r = 0
    for i in n_idx:
        p_range = p_idx[p_idx_k[i]]
        idx.append(range(p_range[0], p_range[1]))
        size = p_range[1] - p_range[0]
        sub_p_idx[p_idx_k[i]] = np.array([prev_r, (prev_r + size)])
        prev_r += size
    return Subset(dataset, np.concatenate(idx)), sub_p_idx


# gen must be a torch generator. If shuffle is false, first training, then validation, then test sets are filled in the
# order that participants appear in the dataset.
def participant_train_val_test_split(
    dataset, train_n, val_n, test_n, shuffle=True, gen=None
):
    assert dataset.p_num() == (
        train_n + val_n + test_n
    ), "The training, validation, and test sets must add up to the total amount of participants!"
    if shuffle:
        idx = torch.randperm(dataset.p_num(), generator=gen)
    else:
        idx = np.arange(dataset.p_num())

    tr_idx = idx[0:train_n]
    v_idx = idx[train_n : (train_n + val_n)]
    te_idx = idx[(train_n + val_n) : (train_n + val_n + test_n)]

    return (
        create_participant_subset(dataset, tr_idx),
        create_participant_subset(dataset, v_idx),
        create_participant_subset(dataset, te_idx),
    )


class FewShotBatchSampler(object):

    def __init__(
        self,
        participants,
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
            N_way - Number of participants to sample per batch.
            K_shot - Number of examples to sample per participant in the batch.
            include_query - If True, returns batch of size N_way*K_shot*2, which
                            can be split into support and query set. Simplifies
                            the implementation of sampling the same participants but
                            distinct examples for support and query set.
            shuffle - If True, examples and participants are newly shuffled in each
                      iteration (for training)
            shuffle_once - If True, examples and participants are shuffled once in
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
        )  # Number of overall participants per batch. The last batch might be smaller.

        # Organize examples by participant
        self.participants = participants
        self.num_participants = len(self.participants)
        self.indices_per_participant = {}
        self.batches_per_participant = (
            {}
        )  # Number of K-shot batches that each participant can provide
        for k, v in self.participants.items():
            self.indices_per_participant[k] = np.arange(v[0], v[1])
            size = self.indices_per_participant[k].shape[0]
            self.batches_per_participant[k] = int(size // self.K_shot)

        # Create a list of participants from which we select the N participants per batch
        self.i_size = sum(self.batches_per_participant.values())
        self.iterations = int(self.i_size // self.N_way)
        self.participant_list = [
            c for c in self.participants for _ in range(self.batches_per_participant[c])
        ]
        if shuffle_once or self.shuffle:
            self.shuffle_data()
        else:
            # For testing, we iterate over participants instead of shuffling them
            sort_idxs = [
                i + p * self.num_participants
                for i, c in enumerate(self.participants)
                for p in range(self.batches_per_participant[c])
            ]
            self.participant_list = np.array(self.participant_list)[
                np.argsort(sort_idxs)
            ].tolist()

    def shuffle_data(self):
        # Shuffle the examples per participant
        for k in self.participants:
            perm = torch.randperm(
                self.indices_per_participant[k].shape[0], generator=self.gen
            )
            self.indices_per_participant[k] = self.indices_per_participant[k][perm]
        # Shuffle the class list from which we sample. Note that this way of shuffling
        # does not prevent to choose the same class twice in a batch. However, for
        # training and validation, this is not a problem.
        participant_list_np = np.array(self.participant_list)
        self.participant_list = list(
            participant_list_np[
                list(torch.randperm(len(self.participant_list), generator=self.gen))
            ]
        )

    def __iter__(self):
        # Shuffle data
        if self.shuffle:
            self.shuffle_data()

        # Sample few-shot batches
        start_index = defaultdict(int)
        for it in range(self.iterations):
            participant_batch = self.participant_list[
                it * self.N_way : (it + 1) * self.N_way
            ]  # Select N participant for the batch
            index_batch = []
            for (
                c
            ) in (
                participant_batch
            ):  # For each participant, select the next K examples and add them to the batch
                index_batch.extend(
                    self.indices_per_participant[c][
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


# This splits participant batch into support and query set and reshapes by participant. Also converts to jax numpy arrays.
def process_p_batch(batch, N_way, K_shot):
    obs, ts, am, obs2, ts2, am2, lab = batch
    _, seq_length, ob_dim = obs.shape
    s_obs, q_obs = obs.chunk(2, dim=0)
    s_ts, q_ts = ts.chunk(2, dim=0)
    s_am, q_am = am.chunk(2, dim=0)
    s_obs2, q_obs2 = obs2.chunk(2, dim=0)
    s_ts2, q_ts2 = ts2.chunk(2, dim=0)
    s_am2, q_am2 = am2.chunk(2, dim=0)
    s_lab, q_lab = lab.chunk(2, dim=0)

    s_obs, q_obs = jnp.asarray(s_obs.reshape(N_way, K_shot, seq_length, ob_dim)), jnp.asarray(q_obs.reshape(
        N_way, K_shot, seq_length, ob_dim
    ))
    s_ts, q_ts = jnp.asarray(s_ts.reshape(N_way, K_shot, seq_length)), jnp.asarray(q_ts.reshape(
        N_way, K_shot, seq_length
    ))
    s_am, q_am = jnp.asarray(s_am.reshape(N_way, K_shot, seq_length)), jnp.asarray(q_am.reshape(
        N_way, K_shot, seq_length
    ))
    s_obs2, q_obs2 = jnp.asarray(s_obs2.reshape(N_way, K_shot, seq_length, ob_dim)), jnp.asarray(q_obs2.reshape(
        N_way, K_shot, seq_length, ob_dim
    ))
    s_ts2, q_ts2 = jnp.asarray(s_ts2.reshape(N_way, K_shot, seq_length)), jnp.asarray(q_ts2.reshape(
        N_way, K_shot, seq_length
    ))
    s_am2, q_am2 = jnp.asarray(s_am2.reshape(N_way, K_shot, seq_length)), jnp.asarray(q_am2.reshape(
        N_way, K_shot, seq_length
    ))
    s_lab, q_lab = jnp.asarray(s_lab.reshape(N_way, K_shot)), jnp.asarray(q_lab.reshape(N_way, K_shot))

    train_batch = (s_obs, s_ts, s_am, s_obs2, s_ts2, s_am2, s_lab)
    val_batch = (s_obs, s_ts, s_am, s_obs2, s_ts2, s_am2, s_lab)
    return train_batch, val_batch
