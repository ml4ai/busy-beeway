import d4rl
import gym
import numpy as np
import os
import pickle
from tqdm import tqdm
import h5py


def qlearning_ant_dataset(env, dataset=None, terminate_on_end=False, **kwargs):
    """
    Returns datasets formatted for use by standard Q-learning algorithms,
    with observations, actions, next_observations, rewards, and a terminal
    flag.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        terminate_on_end (bool): Set done=True on the last timestep
            in a trajectory. Default is False, and will discard the
            last timestep in each trajectory.
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of "done" or episode termination flags.
    """
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset["rewards"].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []
    goal_ = []
    xy_ = []
    done_bef_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if "timeouts" in dataset:
        use_timeouts = True

    episode_step = 0
    for i in range(N - 1):
        obs = dataset["observations"][i].astype(np.float32)
        new_obs = dataset["observations"][i + 1].astype(np.float32)
        action = dataset["actions"][i].astype(np.float32)
        reward = dataset["rewards"][i].astype(np.float32)
        done_bool = bool(dataset["terminals"][i])
        goal = dataset["infos/goal"][i].astype(np.float32)
        xy = dataset["infos/qpos"][i][:2].astype(np.float32)

        if use_timeouts:
            final_timestep = dataset["timeouts"][i]
            next_final_timestep = dataset["timeouts"][i + 1]
        else:
            final_timestep = episode_step == env._max_episode_steps - 1
            next_final_timestep = episode_step == env._max_episode_steps - 2

        done_bef = bool(next_final_timestep)

        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue
        if done_bool or final_timestep:
            episode_step = 0

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        goal_.append(goal)
        xy_.append(xy)
        done_bef_.append(done_bef)
        episode_step += 1

    return {
        "observations": np.array(obs_),
        "actions": np.array(action_),
        "next_observations": np.array(next_obs_),
        "rewards": np.array(reward_),
        "terminals": np.array(done_),
        "goals": np.array(goal_),
        "xys": np.array(xy_),
        "dones_bef": np.array(done_bef_),
    }


def new_get_trj_idx(env, terminate_on_end=False, **kwargs):

    if not hasattr(env, "get_dataset"):
        dataset = kwargs["dataset"]
    else:
        dataset = env.get_dataset()
    N = dataset["rewards"].shape[0]

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if "timeouts" in dataset:
        use_timeouts = True

    episode_step = 0
    start_idx, data_idx = 0, 0
    trj_idx_list = []
    for i in range(N - 1):
        if env.spec and "maze" in env.spec.id:
            done_bool = sum(dataset["infos/goal"][i + 1] - dataset["infos/goal"][i]) > 0
        else:
            done_bool = bool(dataset["terminals"][i])
        if use_timeouts:
            final_timestep = dataset["timeouts"][i]
        else:
            final_timestep = episode_step == env._max_episode_steps - 1
        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            trj_idx_list.append([start_idx, data_idx - 1])
            start_idx = data_idx
            continue
        if done_bool or final_timestep:
            episode_step = 0
            trj_idx_list.append([start_idx, data_idx])
            start_idx = data_idx + 1

        episode_step += 1
        data_idx += 1

    trj_idx_list.append([start_idx, data_idx])

    return trj_idx_list


def load_queries_with_indices(
    env,
    dataset,
    num_query,
    len_query,
    label_type,
    saved_indices,
    saved_labels,
    balance=False,
    scripted_teacher=False,
):

    trj_idx_list = new_get_trj_idx(env, dataset=dataset)  # get_nonmdp_trj_idx(env)

    # to-do: parallel implementation
    trj_idx_list = np.array(trj_idx_list)
    trj_len_list = trj_idx_list[:, 1] - trj_idx_list[:, 0] + 1

    assert max(trj_len_list) > len_query

    total_reward_seq_1, total_reward_seq_2 = np.zeros((num_query, len_query)), np.zeros(
        (num_query, len_query)
    )

    observation_dim = dataset["observations"].shape[-1]
    action_dim = dataset["actions"].shape[-1]

    total_obs_seq_1, total_obs_seq_2 = np.zeros(
        (num_query, len_query, observation_dim)
    ), np.zeros((num_query, len_query, observation_dim))
    total_next_obs_seq_1, total_next_obs_seq_2 = np.zeros(
        (num_query, len_query, observation_dim)
    ), np.zeros((num_query, len_query, observation_dim))
    total_act_seq_1, total_act_seq_2 = np.zeros(
        (num_query, len_query, action_dim)
    ), np.zeros((num_query, len_query, action_dim))
    total_timestep_1, total_timestep_2 = np.zeros(
        (num_query, len_query), dtype=np.int32
    ), np.zeros((num_query, len_query), dtype=np.int32)

    if saved_labels is None:
        query_range = np.arange(num_query)
    else:
        query_range = np.arange(len(saved_labels) - num_query, len(saved_labels))

    for query_count, i in enumerate(
        tqdm(query_range, desc="get queries from saved indices")
    ):
        temp_count = 0
        while temp_count < 2:
            start_idx = saved_indices[temp_count][i]
            end_idx = start_idx + len_query

            reward_seq = dataset["rewards"][start_idx:end_idx]
            obs_seq = dataset["observations"][start_idx:end_idx]
            next_obs_seq = dataset["next_observations"][start_idx:end_idx]
            act_seq = dataset["actions"][start_idx:end_idx]
            timestep_seq = np.arange(1, len_query + 1)

            if temp_count == 0:
                total_reward_seq_1[query_count] = reward_seq
                total_obs_seq_1[query_count] = obs_seq
                total_next_obs_seq_1[query_count] = next_obs_seq
                total_act_seq_1[query_count] = act_seq
                total_timestep_1[query_count] = timestep_seq
            else:
                total_reward_seq_2[query_count] = reward_seq
                total_obs_seq_2[query_count] = obs_seq
                total_next_obs_seq_2[query_count] = next_obs_seq
                total_act_seq_2[query_count] = act_seq
                total_timestep_2[query_count] = timestep_seq

            temp_count += 1

    seg_reward_1 = total_reward_seq_1.copy()
    seg_reward_2 = total_reward_seq_2.copy()

    seg_obs_1 = total_obs_seq_1.copy()
    seg_obs_2 = total_obs_seq_2.copy()

    seg_next_obs_1 = total_next_obs_seq_1.copy()
    seg_next_obs_2 = total_next_obs_seq_2.copy()

    seq_act_1 = total_act_seq_1.copy()
    seq_act_2 = total_act_seq_2.copy()

    seq_timestep_1 = total_timestep_1.copy()
    seq_timestep_2 = total_timestep_2.copy()

    if label_type == 0:  # perfectly rational
        sum_r_t_1 = np.sum(seg_reward_1, axis=1)
        sum_r_t_2 = np.sum(seg_reward_2, axis=1)
        binary_label = 1 * (sum_r_t_1 < sum_r_t_2)
        rational_labels = np.zeros((len(binary_label), 2))
        rational_labels[np.arange(binary_label.size), binary_label] = 1.0
    elif label_type == 1:
        sum_r_t_1 = np.sum(seg_reward_1, axis=1)
        sum_r_t_2 = np.sum(seg_reward_2, axis=1)
        binary_label = 1 * (sum_r_t_1 < sum_r_t_2)
        rational_labels = np.zeros((len(binary_label), 2))
        rational_labels[np.arange(binary_label.size), binary_label] = 1.0
        margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) <= 0).reshape(-1)
        rational_labels[margin_index] = 0.5

    batch = {}
    if scripted_teacher:
        # counter part of human label for comparing with human label.
        batch["labels"] = rational_labels
    else:
        human_labels = np.zeros((len(saved_labels), 2))
        human_labels[np.array(saved_labels) == 0, 0] = 1.0
        human_labels[np.array(saved_labels) == 1, 1] = 1.0
        human_labels[np.array(saved_labels) == -1] = 0.5
        human_labels = human_labels[query_range]
        batch["labels"] = human_labels
    batch["script_labels"] = rational_labels

    batch["observations"] = seg_obs_1  # for compatibility, remove "_1"
    batch["next_observations"] = seg_next_obs_1
    batch["actions"] = seq_act_1
    batch["observations_2"] = seg_obs_2
    batch["next_observations_2"] = seg_next_obs_2
    batch["actions_2"] = seq_act_2
    batch["timestep_1"] = seq_timestep_1
    batch["timestep_2"] = seq_timestep_2
    batch["start_indices"] = saved_indices[0]
    batch["start_indices_2"] = saved_indices[1]

    if balance:
        nonzero_condition = np.any(batch["labels"] != [0.5, 0.5], axis=1)
        (nonzero_idx,) = np.where(nonzero_condition)
        (zero_idx,) = np.where(np.logical_not(nonzero_condition))
        selected_zero_idx = np.random.choice(zero_idx, len(nonzero_idx))
        for key, val in batch.items():
            batch[key] = val[np.concatenate([selected_zero_idx, nonzero_idx])]
        print(f"size of batch after balancing: {len(batch['labels'])}")

    return batch


env = "antmaze-large-play-v2"

gym_env = gym.make(env)

dataset = qlearning_ant_dataset(gym_env)

base_path = "./../ant_labels/antmaze-large-play-v2"

human_indices_2_file, human_indices_1_file, human_labels_file = sorted(
    os.listdir(base_path)
)
with open(os.path.join(base_path, human_indices_1_file), "rb") as fp:  # Unpickling
    human_indices = pickle.load(fp)
with open(os.path.join(base_path, human_indices_2_file), "rb") as fp:  # Unpickling
    human_indices_2 = pickle.load(fp)
with open(os.path.join(base_path, human_labels_file), "rb") as fp:  # Unpickling
    human_labels = pickle.load(fp)

pref_dataset = load_queries_with_indices(
    gym_env,
    dataset,
    1000,
    100,
    label_type=1,
    saved_indices=[human_indices, human_indices_2],
    saved_labels=human_labels,
    balance=False,
    scripted_teacher=False,
)

with h5py.File(base_path + "/antmaze-large-play-v2_pref.hdf5", "a") as f:
    f.create_dataset("states", data=pref_dataset["observations"])
    f.create_dataset("states_2", data=pref_dataset["observations_2"])

    f.create_dataset("actions", data=pref_dataset["actions"])
    f.create_dataset("actions_2", data=pref_dataset["actions_2"])

    f.create_dataset("timesteps", data=pref_dataset["timestep_1"] - 1)
    f.create_dataset("timesteps_2", data=pref_dataset["timestep_2"] - 1)

    am = np.ones((1000, 100)) * 1.0
    am_2 = np.ones((1000, 100)) * 1.0
    f.create_dataset("attn_mask", data=am)
    f.create_dataset("attn_mask_2", data=am_2)

    f.create_dataset("labels", data=pref_dataset["labels"])
