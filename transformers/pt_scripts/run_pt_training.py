import copy
import os
import os.path as osp
import random
import sys
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pyrallis
import torch
import torch.multiprocessing as multiprocessing
import wandb
from flax import nnx
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, random_split
from tqdm import tqdm

import h5py

sys.path.insert(0, os.path.abspath("../.."))
from transformers.data_utils.data_loader import (
    Pref_H5Dataset,
    fast_loader,
    sorted_random_split,
)
from transformers.models.pref_transformer import PT
from transformers.training.logging_utils import logger, setup_logger
from transformers.training.training import PrefTransformerTrainer
from transformers.training.utils import Timer, save_model


@dataclass
class TrainConfig:
    # wandb params
    project: str = "PT-training"
    group: str = "PT"
    name: str = "pt"
    # model params
    embd_dim: int = 256
    pref_attn_embd_dim: Optional[int] = None
    num_heads: int = 4
    attn_dropout: float = 0.1
    resid_dropout: float = 0.1
    intermediate_dim: Optional[int] = None
    num_layers: int = 1
    embd_dropout: float = 0.1
    model_eps: float = 0.1
    max_ep_length: Optional[int] = None
    default_max_pos: int = 2048
    # training params
    dataset_id: str = "D4RL_pen-v2"
    dataset: str = "~/busy-beeway/transformers/pen_labels/AdroitHandPen-v1_pref.hdf5"
    training_split: float = 0.7
    epochs: int = 10
    batch_size: int = 256  # Batch size for all networks
    initial_lr: float = 1e-4
    peak_lr: float = 1e-4 * 10
    end_lr: float = 1e-4
    warmup_steps: Optional[int] = None
    decay_steps: Optional[int] = None
    # evaluation params
    eval_every: int = 1  # How often (time steps) we evaluate
    workers: int = 2
    criteria_type: str = "acc"
    criteria_key: str = "eval_loss"
    # general params
    pin_memory: bool = True
    seed: int = 0
    checkpoints_path: Optional[str] = "~/busy-beeway/transformers"  # Save path

    def __post_init__(self):
        self.name = f"{self.name}-{self.dataset_id}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(
                osp.expanduser(self.checkpoints_path), self.name
            )
        if self.pref_attn_embd_dim is None:
            self.pref_attn_embd_dim = self.embd_dim
        if self.intermediate_dim is None:
            self.intermediate_dim = 4 * self.embd_dim


@pyrallis.wrap()
def train(config: TrainConfig):
    wandb.init(
        config=asdict(config),
        project=config.project,
        group=config.group,
        name=config.name,
        id=str(uuid.uuid4()),
        save_code=True,
    )
    multiprocessing.set_start_method("forkserver")
    data = osp.expanduser(config.dataset)
    try:
        with h5py.File(data, "r") as f:
            if config.max_ep_length is None:
                mep = np.max([np.max(f["timesteps"][:]), np.max(f["timesteps_2"][:])])
            else:
                mep = config.max_ep_length
        data = Pref_H5Dataset(data, mep)
        checkpointer = ocp.Checkpointer(ocp.CompositeCheckpointHandler())
        setup_logger(
            variant=None,
            seed=config.seed,
            base_log_dir=config.checkpoints_path,
            include_exp_prefix_sub_dir=False,
        )

        state_shape, action_shape = data.shapes()
        _, query_len, state_dim = state_shape
        action_dim = action_shape[2]
        rng_key = jax.random.key(config.seed)
        rng_key, rng_subkey = jax.random.split(rng_key, 2)
        t_keys = jax.random.randint(rng_subkey, 2, 0, 10000)
        torch.manual_seed(int(t_keys[0]))
        training_data, test_data = sorted_random_split(
            data, [config.training_split, 1 - config.training_split]
        )
        training_data_loader = fast_loader(
            training_data,
            batch_size=config.batch_size,
            num_workers=config.workers,
            pin_memory=config.pin_memory,
        )
        test_data_loader = fast_loader(
            test_data,
            batch_size=config.batch_size,
            num_workers=config.workers,
            pin_memory=config.pin_memory,
        )

        interval = len(training_data) / config.batch_size
        if int(interval) < interval:
            interval = int(interval + 1)
        else:
            interval = int(interval)

        eval_interval = len(test_data) / config.batch_size
        if int(eval_interval) < eval_interval:
            eval_interval = int(eval_interval + 1)
        else:
            eval_interval = int(eval_interval)
        rng_subkey1, rng_subkey2, rng_subkey3 = jax.random.split(rng_key, 3)
        rngs = nnx.Rngs(rng_subkey1, params=rng_subkey2, dropout=rng_subkey3)
        max_pos = config.default_max_pos
        while query_len > max_pos:
            max_pos *= 2
        model_args = [
            state_dim,
            action_dim,
            mep,
            config.embd_dim,
            config.pref_attn_embd_dim,
            config.num_heads,
            config.attn_dropout,
            config.resid_dropout,
            config.intermediate_dim,
            config.num_layers,
            config.embd_dropout,
            max_pos,
            config.model_eps,
            config.seed,
        ]

        trans = PT(
            state_dim=model_args[0],
            action_dim=model_args[1],
            max_episode_steps=model_args[2],
            embd_dim=model_args[3],
            pref_attn_embd_dim=model_args[4],
            num_heads=model_args[5],
            attn_dropout=model_args[6],
            resid_dropout=model_args[7],
            intermediate_dim=model_args[8],
            num_layers=model_args[9],
            embd_dropout=model_args[10],
            max_pos=model_args[11],
            eps=model_args[12],
            rngs=rngs,
        )

        model_args = np.array(model_args)

        if config.warmup_steps is None:
            warmup_steps = int(config.epochs * interval * 0.05)
        else:
            warmup_steps = config.warmup_steps

        if config.decay_steps is None:
            decay_steps = int(config.epochs * interval)
        else:
            decay_steps = config.decay_steps

        model = PrefTransformerTrainer(
            trans,
            init_value=config.initial_lr,
            peak_value=config.peak_lr,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=config.end_lr,
        )
        c_best_epoch = 0
        if config.criteria_type == "acc":
            c_criteria_key = -np.inf
        else:
            c_criteria_key = np.inf

        batch_keys = [
            "states",
            "actions",
            "timesteps",
            "attn_mask",
            "states_2",
            "actions_2",
            "timesteps_2",
            "attn_mask_2",
            "labels",
        ]
        for epoch in range(config.epochs + 1):
            metrics = {
                "train_time": np.nan,
                "training_loss": [],
                "training_acc": [],
                "eval_loss": [],
                "eval_acc": [],
                "best_epoch": c_best_epoch,
                f"{config.criteria_key}_best": c_criteria_key,
            }
            if epoch:
                with Timer() as train_timer:
                    for i, t_data in tqdm(
                        enumerate(training_data_loader),
                        total=interval,
                        desc=f"Training Epoch {epoch}",
                    ):
                        batch = {}
                        for k, dat in enumerate(t_data):
                            batch[batch_keys[k]] = jnp.asarray(dat)
                        for key, val in model.train(batch).items():
                            metrics[key].append(val)
                        del batch
                metrics["train_time"] = train_timer()
            else:
                # for using early stopping with train loss.
                metrics["training_loss"] = np.nan

            # eval phase
            if epoch % config.eval_every == 0:
                for j, e_data in tqdm(
                    enumerate(test_data_loader),
                    total=eval_interval,
                    desc=f"Evaluation Epoch {epoch}",
                ):
                    batch = {}
                    for k, dat in enumerate(e_data):
                        batch[batch_keys[k]] = jnp.asarray(dat)
                    for key, val in model.evaluation(batch).items():
                        metrics[key].append(val)
                    del batch
                criteria = np.mean(metrics[config.criteria_key])

                if config.criteria_type == "acc":
                    if criteria >= c_criteria_key:
                        c_best_epoch = epoch
                        c_criteria_key = criteria
                        metrics["best_epoch"] = c_best_epoch
                        metrics[f"{config.criteria_key}_best"] = c_criteria_key
                        if config.checkpoints_path is not None:
                            save_model(
                                trans,
                                model_args,
                                "best_model",
                                config.checkpoints_path,
                                checkpointer,
                            )
                else:
                    if criteria <= c_criteria_key:
                        c_best_epoch = epoch
                        c_criteria_key = criteria
                        metrics["best_epoch"] = c_best_epoch
                        metrics[f"{config.criteria_key}_best"] = c_criteria_key
                        if config.checkpoints_path is not None:
                            save_model(
                                trans,
                                model_args,
                                "best_model",
                                config.checkpoints_path,
                                checkpointer,
                            )
            for key, val in metrics.items():
                if isinstance(val, list):
                    if len(val):
                        metrics[key] = np.mean(val)
                    else:
                        metrics[key] = np.nan
            wandb.log(metrics, step=epoch)
            logger.record_dict(metrics | {"epoch": epoch})
            logger.dump_tabular(with_prefix=False, with_timestamp=False)
        if config.checkpoints_path is not None:
            save_model(
                trans, model_args, "model", config.checkpoints_path, checkpointer
            )
        checkpointer.close()
    except FileNotFoundError:
        raise FileNotFoundError(f"{data} not found!")
    sys.exit(0)


if __name__ == "__main__":
    train()
