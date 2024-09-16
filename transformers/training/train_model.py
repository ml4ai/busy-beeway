import os.path as osp

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.early_stopping import EarlyStopping
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from transformers.models.intervention_mlp import MLP
from transformers.models.pref_transformer import PT
from transformers.training.jax_utils import batch_to_jax
from transformers.training.logging_utils import logger, setup_logger
from transformers.training.training import (
    InterventionMLPTrainer,
    PrefTransformerTrainer,
)
from transformers.training.utils import Timer, save_pickle


def train_pt(
    data,
    seed,
    train_split=0.7,
    batch_size=64,
    num_workers=2,
    n_epochs=50,
    eval_period=1,
    do_early_stop=False,
    criteria_key="eval_loss",
    save_dir="~/busy-beeway/transformers/logs",
    save_model=True,
    **kwargs,
):

    save_dir = osp.expanduser(save_dir)
    setup_logger(
        variant=None,
        seed=seed,
        base_log_dir=save_dir,
        include_exp_prefix_sub_dir=False,
    )

    _, query_len, observation_dim = data.obs_shape()
    rng_key = jax.random.PRNGKey(seed)
    rng_key, rng_subkey = jax.random.split(rng_key, 2)
    gen1 = torch.Generator().manual_seed(int(rng_subkey[0]))
    gen2 = torch.Generator().manual_seed(int(rng_subkey[1]))
    training_data, test_data = random_split(
        data, [train_split, 1 - train_split], generator=gen1
    )
    training_data_loader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=gen2,
        pin_memory=True,
    )
    test_data_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )

    interval = len(training_data_loader)
    eval_interval = len(test_data_loader)
    max_pos = 512
    while query_len > max_pos:
        max_pos *= 2
    embd_dim = kwargs.get("embd_dim", min(batch_size, 256))
    trans = PT(
        observation_dim=observation_dim,
        max_episode_steps=kwargs.get("max_episode_steps", 1219),
        embd_dim=embd_dim,
        pref_attn_embd_dim=kwargs.get("pref_attn_embd_dim", embd_dim),
        num_heads=kwargs.get("num_heads", 4),
        attn_dropout=kwargs.get("attn_dropout", 0.1),
        resid_dropout=kwargs.get("resid_dropout", 0.1),
        intermediate_dim=kwargs.get("intermediate_dim", 4 * embd_dim),
        num_layers=kwargs.get("num_layers", 1),
        embd_dropout=kwargs.get("embd_dropout", 0.1),
        max_pos=kwargs.get("max_pos", max_pos),
        eps=kwargs.get("eps", 0.1),
    )

    rng_subkey1, rng_subkey2, rng_subkey3 = jax.random.split(rng_key, 3)
    model = PrefTransformerTrainer(
        trans,
        rng_subkey1,
        rng_subkey2,
        init_value=kwargs.get("init_value", 0),
        peak_value=kwargs.get("peak_value", 1e-4),
        warmup_steps=kwargs.get("warmup_steps", int(n_epochs * interval * 0.1)),
        decay_steps=kwargs.get("decay_steps", int(n_epochs * interval)),
        end_value=kwargs.get("end_value", 0),
    )
    early_stop = EarlyStopping(min_delta=1e-3, patience=10)
    c_best_epoch = np.nan
    c_criteria_key = np.nan
    for epoch, (s_key, t_key, e_key) in enumerate(
        jax.random.split(rng_subkey3, (n_epochs + 1, 3))
    ):
        metrics = {
            "epoch": epoch,
            "train_time": np.nan,
            "training_loss": [],
            "eval_loss": [],
            "best_epoch": c_best_epoch,
            f"{criteria_key}_best": c_criteria_key,
        }
        if epoch:
            with Timer() as train_timer:
                t_keys = jax.random.split(t_key, interval)
                for i, t_data in tqdm(
                    enumerate(training_data_loader),
                    total=interval,
                    desc=f"Training Epoch {epoch}",
                ):
                    batch = {}
                    (
                        batch["observations"],
                        batch["timesteps"],
                        batch["attn_mask"],
                        batch["observations_2"],
                        batch["timesteps_2"],
                        batch["attn_mask_2"],
                        batch["labels"],
                    ) = t_data
                    for k in batch:
                        batch[k] = jnp.asarray(batch[k])
                    batch = batch_to_jax(batch)
                    for key, val in model.train(batch, t_keys[i]).items():
                        metrics[key].append(val)
            metrics["train_time"] = train_timer()
        else:
            # for using early stopping with train loss.
            metrics["training_loss"] = np.nan

        # eval phase
        if epoch % eval_period == 0:
            e_keys = jax.random.split(e_key, eval_interval)
            for j, e_data in tqdm(
                enumerate(test_data_loader),
                total=eval_interval,
                desc=f"Evaluation Epoch {epoch}",
            ):
                batch = {}
                (
                    batch["observations"],
                    batch["timesteps"],
                    batch["attn_mask"],
                    batch["observations_2"],
                    batch["timesteps_2"],
                    batch["attn_mask_2"],
                    batch["labels"],
                ) = e_data
                for k in batch:
                    batch[k] = jnp.asarray(batch[k])
                batch = batch_to_jax(batch)
                for key, val in model.evaluation(batch, e_keys[j]).items():
                    metrics[key].append(val)
            criteria = np.mean(metrics[criteria_key])
            early_stop = early_stop.update(criteria)
            if early_stop.should_stop and do_early_stop:
                for key, val in metrics.items():
                    if isinstance(val, list):
                        if len(val):
                            metrics[key] = np.mean(val)
                        else:
                            metrics[key] = np.nan
                logger.record_dict(metrics)
                logger.dump_tabular(with_prefix=False, with_timestamp=False)
                print("Met early stopping criteria, breaking...")
                break
            elif epoch > 0 and early_stop.has_improved:
                c_best_epoch = epoch
                c_criteria_key = criteria
                metrics["best_epoch"] = c_best_epoch
                metrics[f"{criteria_key}_best"] = c_criteria_key
                save_data = {"model": model, "epoch": epoch}
                save_pickle(save_data, "best_model.pkl", save_dir)

        for key, val in metrics.items():
            if isinstance(val, list):
                if len(val):
                    metrics[key] = np.mean(val)
                else:
                    metrics[key] = np.nan
        logger.record_dict(metrics)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)
    if save_model:
        save_data = {"model": model, "epoch": epoch}
        save_pickle(save_data, "model.pkl", save_dir)


def train_imlp(
    data,
    training_data_idx,
    test_data_idx,
    batch_size=64,
    n_epochs=50,
    eval_period=1,
    do_early_stop=False,
    criteria_key="eval_loss",
    seed=2024,
    save_dir="~/busy-beeway/transformers/logs",
    save_model=True,
    **kwargs,
):

    save_dir = osp.expanduser(save_dir)
    setup_logger(
        variant=None, seed=seed, base_log_dir=save_dir, include_exp_prefix_sub_dir=False
    )
    set_random_seed(seed)
    rng = np.random.default_rng(seed)
    data_size = training_data_idx.shape[0]
    _, observation_dim = data["observations"].shape
    eval_data_size = test_data_idx.shape[0]
    interval = int(data_size / batch_size) + 1
    eval_interval = int(eval_data_size / batch_size) + 1

    imlp = MLP(
        observation_dim=observation_dim,
        embd_dim=kwargs.get("embd_dim", batch_size),
        activation=kwargs.get("activation", "relu"),
        embd_dropout=kwargs.get("embd_dropout", 0.1),
    )
    model = InterventionMLPTrainer(
        imlp,
        init_value=kwargs.get("init_value", 0),
        peak_value=kwargs.get("peak_value", 1e-4),
        warmup_steps=kwargs.get("warmup_steps", int(n_epochs * interval * 0.1)),
        decay_steps=kwargs.get("decay_steps", int(n_epochs * interval)),
        end_value=kwargs.get("end_value", 0),
    )
    early_stop = EarlyStopping(min_delta=1e-3, patience=10)
    c_best_epoch = np.nan
    c_criteria_key = np.nan
    for epoch in range(n_epochs + 1):
        metrics = {
            "epoch": epoch,
            "train_time": np.nan,
            "training_loss": [],
            "training_acc": [],
            "eval_loss": [],
            "eval_acc": [],
            "best_epoch": c_best_epoch,
            f"{criteria_key}_best": c_criteria_key,
        }
        if epoch:
            # train phase
            shuffled_idx = rng.permutation(data_size)
            for i in range(interval):
                start_pt = i * batch_size
                end_pt = min((i + 1) * batch_size, data_size)
                with Timer() as train_timer:
                    # train
                    batch = batch_to_jax(
                        index_batch(
                            data, training_data_idx[shuffled_idx[start_pt:end_pt]]
                        )
                    )
                    for key, val in model.train(batch).items():
                        metrics[key].append(val)
            metrics["train_time"] = train_timer()
        else:
            # for using early stopping with train loss.
            metrics["training_loss"] = np.nan
            metrics["training_acc"] = np.nan

        # eval phase
        if epoch % eval_period == 0:
            for j in range(eval_interval):
                eval_start_pt, eval_end_pt = j * batch_size, min(
                    (j + 1) * batch_size, eval_data_size
                )
                # batch_eval = batch_to_jax(index_batch(pref_eval_dataset, range(eval_start_pt, eval_end_pt)))
                batch_eval = batch_to_jax(
                    index_batch(data, test_data_idx[eval_start_pt:eval_end_pt])
                )
                for key, val in model.evaluation(batch_eval).items():
                    metrics[key].append(val)
            criteria = np.mean(metrics[criteria_key])
            early_stop = early_stop.update(criteria)
            if early_stop.should_stop and do_early_stop:
                for key, val in metrics.items():
                    if isinstance(val, list):
                        if len(val):
                            metrics[key] = np.mean(val)
                        else:
                            metrics[key] = np.nan
                logger.record_dict(metrics)
                logger.dump_tabular(with_prefix=False, with_timestamp=False)
                print("Met early stopping criteria, breaking...")
                break
            elif epoch > 0 and early_stop.has_improved:
                c_best_epoch = epoch
                c_criteria_key = criteria
                metrics["best_epoch"] = c_best_epoch
                metrics[f"{criteria_key}_best"] = c_criteria_key
                save_data = {"model": model, "epoch": epoch}
                save_pickle(save_data, "best_model.pkl", save_dir)

        for key, val in metrics.items():
            if isinstance(val, list):
                if len(val):
                    metrics[key] = np.mean(val)
                else:
                    metrics[key] = np.nan
        logger.record_dict(metrics)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)
    if save_model:
        save_data = {"model": model, "epoch": epoch}
        save_pickle(save_data, "model.pkl", save_dir)
