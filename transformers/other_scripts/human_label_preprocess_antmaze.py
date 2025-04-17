import argparse
import os
import sys

import h5py
import minari
import mujoco
import numpy as np

sys.path.insert(0, os.path.abspath("../.."))
from pathlib import Path

import imageio
from argformat import StructuredFormatter
from PIL import Image, ImageDraw
from tqdm import tqdm, trange


def visualize_query(
    gym_env,
    dataset,
    batch,
    query_len,
    num_query,
    width=500,
    height=500,
    save_dir="./video",
    verbose=False,
    slow=False,
):
    save_dir = os.path.join(save_dir, gym_env.spec.id)
    if slow:
        save_dir = os.path.join(save_dir, "slow")
    os.makedirs(save_dir, exist_ok=True)

    gym_env.unwrapped.ant_env.mujoco_renderer.width = width
    gym_env.unwrapped.ant_env.mujoco_renderer.height = height
    gym_env.unwrapped.ant_env.mujoco_renderer.default_cam_config = {
        "distance": 40,
        "elevation": -65,
        "lookat": [0, 0, 0],
    }

    for seg_idx in trange(num_query):
        frames = []
        frames_2 = []

        gym_env.unwrapped.ant_env.mujoco_renderer.camera_id = -1
        gym_env.reset()
        gym_env.unwrapped.ant_env.reset()

        gym_env.unwrapped.ant_env.set_state(
            batch["states"][seg_idx, 0, 2:17], batch["states"][seg_idx, 0, 17:]
        )
        gym_env.unwrapped.goal = batch["goals"][seg_idx]
        gym_env.unwrapped.update_target_site_pos()
        for t in trange(query_len, leave=False):
            gym_env.unwrapped.ant_env.set_state(
                batch["states"][seg_idx, t, 2:17], batch["states"][seg_idx, t, 17:]
            )

            curr_frame = gym_env.render()
            if slow:
                frame_img = Image.fromarray(curr_frame)
                draw = ImageDraw.Draw(frame_img)
                draw.text((width - 10, 0), f"{t + 1}", fill="black")
                draw.text((0, 0), "0", fill="black")
                curr_frame = np.asarray(frame_img)
            for i in range(10):
                frames.append(curr_frame)

        gym_env.unwrapped.ant_env.mujoco_renderer.camera_id = 0
        gym_env.reset()
        gym_env.unwrapped.ant_env.reset()

        gym_env.unwrapped.ant_env.set_state(
            batch["states"][seg_idx, 0, 2:17], batch["states"][seg_idx, 0, 17:]
        )
        gym_env.unwrapped.goal = batch["goals"][seg_idx]
        gym_env.unwrapped.update_target_site_pos()
        for t in trange(query_len, leave=False):
            gym_env.unwrapped.ant_env.set_state(
                batch["states"][seg_idx, t, 2:17], batch["states"][seg_idx, t, 17:]
            )

            curr_frame = gym_env.render()
            if slow:
                frame_img = Image.fromarray(curr_frame)
                draw = ImageDraw.Draw(frame_img)
                draw.text((width - 10, 0), f"{t + 1}", fill="black")
                draw.text((0, 0), "0", fill="black")
                curr_frame = np.asarray(frame_img)
            for i in range(10):
                frames.append(curr_frame)

        gym_env.unwrapped.ant_env.mujoco_renderer.camera_id = -1
        gym_env.reset()
        gym_env.unwrapped.ant_env.reset()
        gym_env.unwrapped.ant_env.set_state(
            batch["states_2"][seg_idx, 0, 2:17], batch["states_2"][seg_idx, 0, 17:]
        )
        gym_env.unwrapped.goal = batch["goals_2"][seg_idx]
        gym_env.unwrapped.update_target_site_pos()
        for t in trange(query_len, leave=False):
            gym_env.unwrapped.ant_env.set_state(
                batch["states_2"][seg_idx, t, 2:17], batch["states_2"][seg_idx, t, 17:]
            )

            curr_frame = gym_env.render()
            if slow:
                frame_img = Image.fromarray(curr_frame)
                draw = ImageDraw.Draw(frame_img)
                draw.text((width - 10, 0), f"{t + 1}", fill="black")
                draw.text((0, 0), "1", fill="black")
                curr_frame = np.asarray(frame_img)
                curr_frame = np.asarray(frame_img)
            for i in range(10):
                frames_2.append(curr_frame)

        gym_env.unwrapped.ant_env.mujoco_renderer.camera_id = 0
        gym_env.reset()
        gym_env.unwrapped.ant_env.reset()
        gym_env.unwrapped.ant_env.set_state(
            batch["states_2"][seg_idx, 0, 2:17], batch["states_2"][seg_idx, 0, 17:]
        )
        gym_env.unwrapped.goal = batch["goals_2"][seg_idx]
        gym_env.unwrapped.update_target_site_pos()
        for t in trange(query_len, leave=False):
            gym_env.unwrapped.ant_env.set_state(
                batch["states_2"][seg_idx, t, 2:17], batch["states_2"][seg_idx, t, 17:]
            )

            curr_frame = gym_env.render()
            if slow:
                frame_img = Image.fromarray(curr_frame)
                draw = ImageDraw.Draw(frame_img)
                draw.text((width - 10, 0), f"{t + 1}", fill="black")
                draw.text((0, 0), "1", fill="black")
                curr_frame = np.asarray(frame_img)
                curr_frame = np.asarray(frame_img)
            for i in range(10):
                frames_2.append(curr_frame)

        video = np.concatenate((np.array(frames), np.array(frames_2)), axis=2)

        fps = 3 if slow else 60
        writer = imageio.get_writer(
            os.path.join(save_dir, f"./idx{seg_idx}.mp4"), fps=fps
        )
        for frame in tqdm(video, leave=False):
            writer.append_data(frame)
        writer.close()


def main(argv):
    parser = argparse.ArgumentParser(
        description="Creates Videos of AntMaze for human labeling",
        formatter_class=StructuredFormatter,
    )
    parser.add_argument(
        "-e",
        "--env_name",
        type=str,
        default="D4RL/antmaze/medium-play-v1",
        help="Environment Name",
    )
    parser.add_argument(
        "-s",
        "--save_dir",
        type=str,
        default="~/busy-beeway/transformers/ant_labels/",
        help="Path for saving output.",
    )
    parser.add_argument(
        "-n",
        "--num_query",
        type=int,
        default=100,
        help="Number of Queries",
    )
    parser.add_argument(
        "-q",
        "--query_len",
        type=int,
        default=100,
        help="Length of Query",
    )
    parser.add_argument(
        "-l",
        "--slow",
        action="store_true",
        help="slow option for external feedback",
    )
    parser.add_argument(
        "-r",
        "--random_seed",
        type=int,
        default=138713,
        help="Random Seed",
    )
    args = parser.parse_args(argv)
    save_dir = os.path.expanduser(args.save_dir)
    video_size = {"medium": (500, 500), "large": (600, 450)}

    dataset = minari.load_dataset(args.env_name)
    gym_env = dataset.recover_environment(render_mode="rgb_array")
    if "medium" in args.env_name:
        width, height = video_size["medium"]
    elif "large" in args.env_name:
        width, height = video_size["large"]

    rng = np.random.default_rng(args.random_seed)
    states_1 = []
    states_2 = []
    actions_1 = []
    actions_2 = []
    timesteps_1 = []
    timesteps_2 = []
    attn_mask_1 = []
    attn_mask_2 = []
    goals_1 = []
    goals_2 = []
    idx = rng.choice(dataset.episode_indices,args.num_query * 2,replace=True)
    episodes = dataset.iterate_episodes(idx)
    for i in range(args.num_query):
        episode_1 = next(episodes)
        episode_2 = next(episodes)

        am_1 = np.ones(args.query_len)
        am_2 = np.ones(args.query_len)
        N_1 = (len(episode_1) - episode_1.rewards.sum()) - args.query_len
        N_2 = (len(episode_2) - episode_2.rewards.sum()) - args.query_len
        if N_1 <= 0:
            st_idx_1 = 0
        else:
            st_idx_1 = rng.integers(0, N_1)

        if N_2 <= 0:
            st_idx_2 = 0
        else:
            st_idx_2 = rng.integers(0, N_2)

        end_idx_1 = st_idx_1 + args.query_len
        end_idx_2 = st_idx_2 + args.query_len

        sts_1 = np.concatenate(
            [
                episode_1.observations["desired_goal"][st_idx_1:end_idx_1, ...],
                episode_1.observations["achieved_goal"][st_idx_1:end_idx_1, ...],
                episode_1.observations["observation"][st_idx_1:end_idx_1, ...],
            ],
            axis=1,
        )
        sts_2 = np.concatenate(
            [
                episode_2.observations["desired_goal"][st_idx_2:end_idx_2, ...],
                episode_2.observations["achieved_goal"][st_idx_2:end_idx_2, ...],
                episode_2.observations["observation"][st_idx_2:end_idx_2, ...],
            ],
            axis=1,
        )

        acts_1 = episode_1.actions[st_idx_1:end_idx_1, ...]
        acts_2 = episode_2.actions[st_idx_2:end_idx_2, ...]

        ts_1 = np.arange(st_idx_1, end_idx_1)
        ts_2 = np.arange(st_idx_2, end_idx_2)

        states_1.append(sts_1)
        states_2.append(sts_2)

        actions_1.append(acts_1)
        actions_2.append(acts_2)

        timesteps_1.append(ts_1)
        timesteps_2.append(ts_2)

        attn_mask_1.append(am_1)
        attn_mask_2.append(am_2)

        goals_1.append(episode_1.infos["goal"][0])
        goals_2.append(episode_2.infos["goal"][0])

    batch = {
        "states": np.stack(states_1),
        "actions": np.stack(actions_1),
        "timesteps": np.stack(timesteps_1),
        "attn_mask": np.stack(attn_mask_1),
        "states_2": np.stack(states_2),
        "actions_2": np.stack(actions_2),
        "timesteps_2": np.stack(timesteps_2),
        "attn_mask_2": np.stack(attn_mask_2),
        "goals": np.stack(goals_1),
        "goals_2": np.stack(goals_2),
    }
    with h5py.File(f"{save_dir}/{gym_env.spec.id}_pref.hdf5", "a") as f:
        if "states" in f:
            del f["states"]
        f.create_dataset("states", data=batch["states"], chunks=True)

        if "actions" in f:
            del f["actions"]
        f.create_dataset("actions", data=batch["actions"], chunks=True)

        if "timesteps" in f:
            del f["timesteps"]
        f.create_dataset("timesteps", data=batch["timesteps"], chunks=True)

        if "attn_mask" in f:
            del f["attn_mask"]
        f.create_dataset("attn_mask", data=batch["attn_mask"], chunks=True)

        if "states_2" in f:
            del f["states_2"]
        f.create_dataset("states_2", data=batch["states_2"], chunks=True)

        if "actions_2" in f:
            del f["actions_2"]
        f.create_dataset("actions_2", data=batch["actions_2"], chunks=True)

        if "timesteps_2" in f:
            del f["timesteps_2"]
        f.create_dataset("timesteps_2", data=batch["timesteps_2"], chunks=True)

        if "attn_mask_2" in f:
            del f["attn_mask_2"]
        f.create_dataset("attn_mask_2", data=batch["attn_mask_2"], chunks=True)

    visualize_query(
        gym_env,
        dataset,
        batch,
        args.query_len,
        args.num_query,
        width=width,
        height=height,
        save_dir=save_dir,
        slow=args.slow,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
