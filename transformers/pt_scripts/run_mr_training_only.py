import copy
import os
import os.path as osp
import random
import sys
import uuid
from dataclasses import asdict, dataclass, field
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
    Pref_H5Dataset_from_disk,
    Pref_H5Dataset_from_ram,
    fast_loader,
    sorted_random_split,
)
from transformers.models.q_mlp import Q_MLP
from transformers.training.logging_utils import logger, setup_logger
from transformers.training.training import MRTrainer
from transformers.training.utils import Timer, save_model


@dataclass
class TrainConfig:
    # wandb params
    project: str = "MR-training"
    group: str = "MR"
    name: str = "mr"
    # model params
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    orthogonal_init: bool = False
    activations: str = "relu"
    activation_final: str = "none"
    # training params
    dataset_id: str = "D4RL/pen-v2"
    dataset: str = "~/busy-beeway/transformers/pen_labels/AdroitHandPen-v1_pref.hdf5"
    training_reduce: float = (
        0.0  # artificially reduce training data by the percentage given (default is no reduction)
    )
    epochs: int = 10
    batch_size: int = 256  # Batch size for all networks
    lr: float = 3e-4
    workers: int = 2
    criteria_type: str = "acc"
    criteria_key: str = "training_acc"
    # general params
    pin_memory: bool = True
    seed: int = 0
    data_from_disk: bool = False
    checkpoints_path: Optional[str] = "~/busy-beeway/transformers"  # Save path

    def __post_init__(self):
        self.name = f"{self.name}-{self.dataset_id}-{str(uuid.uuid4())[:8]}-{self.training_reduce}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(
                osp.expanduser(self.checkpoints_path), self.name
            )


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
    if config.data_from_disk:
        try:
            data = Pref_H5Dataset_from_disk(data, -1)
            checkpointer = ocp.Checkpointer(ocp.CompositeCheckpointHandler())
            setup_logger(
                variant=None,
                seed=config.seed,
                base_log_dir=config.checkpoints_path,
                include_exp_prefix_sub_dir=False,
            )

            state_shape, action_shape = data.shapes()
            state_dim = state_shape[2]
            action_dim = action_shape[2]
            rng_key = jax.random.key(config.seed)
            rng_key, rng_subkey = jax.random.split(rng_key, 2)
            t_keys = jax.random.randint(rng_subkey, 2, 0, 10000)
            torch.manual_seed(int(t_keys[0]))

            if config.training_reduce:
                training_data, _ = sorted_random_split(
                    data, [1 - config.training_reduce, config.training_reduce]
                )
            else:
                training_data = data

            training_data_loader = fast_loader(
                training_data,
                batch_size=config.batch_size,
                num_workers=config.workers,
                pin_memory=config.pin_memory,
            )

            interval = len(training_data) / config.batch_size
            if int(interval) < interval:
                interval = int(interval + 1)
            else:
                interval = int(interval)

            rng_subkey1, rng_subkey2, rng_subkey3 = jax.random.split(rng_key, 3)
            rngs = nnx.Rngs(rng_subkey1, params=rng_subkey2, dropout=rng_subkey3)

            model_args = [
                state_dim,
                action_dim,
                config.orthogonal_init,
                config.activations,
                config.activation_final,
                config.seed,
            ] + config.hidden_dims

            qmlp = Q_MLP(
                state_dim=model_args[0],
                action_dim=model_args[1],
                hidden_dims=model_args[6:],
                orthogonal_init=model_args[2],
                activations=model_args[3],
                activation_final=model_args[4],
                rngs=rngs,
            )

            options = [
                "cos",
                "tanh",
                "relu",
                "softplus",
                "sin",
                "leaky_relu",
                "swish",
                "none",
            ]
            for i, j in enumerate(options):
                if model_args[3] == j:
                    model_args[3] = i
                if model_args[4] == j:
                    model_args[4] = i
            model_args = np.array(model_args)

            model = MRTrainer(qmlp, lr=config.lr)
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
                    "best_epoch": c_best_epoch,
                    f"{config.criteria_key}_best": c_criteria_key,
                }

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

                criteria = np.mean(metrics[config.criteria_key])

                if config.criteria_type == "acc":
                    if criteria >= c_criteria_key:
                        c_best_epoch = epoch
                        c_criteria_key = criteria
                        metrics["best_epoch"] = c_best_epoch
                        metrics[f"{config.criteria_key}_best"] = c_criteria_key
                        if config.checkpoints_path is not None:
                            save_model(
                                qmlp,
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
                                qmlp,
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
                    qmlp, model_args, "model", config.checkpoints_path, checkpointer
                )
            checkpointer.close()
        except FileNotFoundError:
            raise FileNotFoundError(f"{data} not found!")
    else:
        try:
            data = Pref_H5Dataset_from_ram(data, -1)
            checkpointer = ocp.Checkpointer(ocp.CompositeCheckpointHandler())
            setup_logger(
                variant=None,
                seed=config.seed,
                base_log_dir=config.checkpoints_path,
                include_exp_prefix_sub_dir=False,
            )

            state_shape, action_shape = data.shapes()
            state_dim = state_shape[2]
            action_dim = action_shape[2]
            rng_key = jax.random.key(config.seed)
            rng_key, rng_subkey = jax.random.split(rng_key, 2)
            t_keys = jax.random.randint(rng_subkey, 2, 0, 10000)
            torch.manual_seed(int(t_keys[0]))

            if config.training_reduce:
                training_data, _ = random_split(
                    data, [1 - config.training_reduce, config.training_reduce]
                )
            else:
                training_data = data

            training_data_loader = DataLoader(
                training_data,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.workers,
                pin_memory=config.pin_memory,
            )

            interval = len(training_data) / config.batch_size
            if int(interval) < interval:
                interval = int(interval + 1)
            else:
                interval = int(interval)

            rng_subkey1, rng_subkey2, rng_subkey3 = jax.random.split(rng_key, 3)
            rngs = nnx.Rngs(rng_subkey1, params=rng_subkey2, dropout=rng_subkey3)

            model_args = [
                state_dim,
                action_dim,
                config.orthogonal_init,
                config.activations,
                config.activation_final,
                config.seed,
            ] + config.hidden_dims

            qmlp = Q_MLP(
                state_dim=model_args[0],
                action_dim=model_args[1],
                hidden_dims=model_args[6:],
                orthogonal_init=model_args[2],
                activations=model_args[3],
                activation_final=model_args[4],
                rngs=rngs,
            )

            options = [
                "cos",
                "tanh",
                "relu",
                "softplus",
                "sin",
                "leaky_relu",
                "swish",
                "none",
            ]
            for i, j in enumerate(options):
                if model_args[3] == j:
                    model_args[3] = i
                if model_args[4] == j:
                    model_args[4] = i
            model_args = np.array(model_args)

            model = MRTrainer(qmlp, lr=config.lr)
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
                    "best_epoch": c_best_epoch,
                    f"{config.criteria_key}_best": c_criteria_key,
                }
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

                criteria = np.mean(metrics[config.criteria_key])

                if config.criteria_type == "acc":
                    if criteria >= c_criteria_key:
                        c_best_epoch = epoch
                        c_criteria_key = criteria
                        metrics["best_epoch"] = c_best_epoch
                        metrics[f"{config.criteria_key}_best"] = c_criteria_key
                        if config.checkpoints_path is not None:
                            save_model(
                                qmlp,
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
                                qmlp,
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
                    qmlp, model_args, "model", config.checkpoints_path, checkpointer
                )
            checkpointer.close()
        except FileNotFoundError:
            raise FileNotFoundError(f"{data} not found!")
    sys.exit(0)


if __name__ == "__main__":
    train()
