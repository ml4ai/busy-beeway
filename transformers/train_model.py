import os.path as osp

import numpy as np
from flax.training.early_stopping import EarlyStopping

from intervention_mlp import MLP
from jax_utils import batch_to_jax
from logging_utils import logger, setup_logger
from pref_transformer import PT
from training import InterventionMLPTrainer, PrefTransformerTrainer
from utils import Timer, index_batch, save_pickle, set_random_seed


def train_pt(
    training_data,
    test_data,
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
    data_size, query_len, observation_dim = training_data["observations"].shape
    eval_data_size = test_data["observations"].shape[0]
    interval = int(data_size / batch_size) + 1
    eval_interval = int(eval_data_size / batch_size) + 1
    trans = PT(
        observation_dim=observation_dim,
        max_episode_steps=kwargs.get("max_episode_steps", 500),
        embd_dim=kwargs.get("embd_dim", batch_size),
        pref_attn_embd_dim=kwargs.get("pref_attn_embd_dim", batch_size),
        num_heads=kwargs.get("num_heads", 4),
        attn_dropout=kwargs.get("attn_dropout", 0.1),
        resid_dropout=kwargs.get("resid_dropout", 0.1),
        intermediate_dim=kwargs.get("intermediate_dim", 4 * batch_size),
        activation=kwargs.get("activation", "relu"),
        num_layers=kwargs.get("num_layers", 1),
        embd_dropout=kwargs.get("embd_dropout", 0.1),
        eps=kwargs.get("eps", 0.1),
    )
    model = PrefTransformerTrainer(
        trans,
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
            "eval_loss": [],
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
                        index_batch(training_data, shuffled_idx[start_pt:end_pt])
                    )
                    for key, val in model.train(batch).items():
                        metrics[key].append(val)
            metrics["train_time"] = train_timer()
        else:
            # for using early stopping with train loss.
            metrics["training_loss"] = np.nan

        # eval phase
        if epoch % eval_period == 0:
            for j in range(eval_interval):
                eval_start_pt, eval_end_pt = j * batch_size, min(
                    (j + 1) * batch_size, eval_data_size
                )
                # batch_eval = batch_to_jax(index_batch(pref_eval_dataset, range(eval_start_pt, eval_end_pt)))
                batch_eval = batch_to_jax(
                    index_batch(test_data, list(range(eval_start_pt, eval_end_pt)))
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


def train_imlp(
    training_data,
    test_data,
    batch_size=64,
    n_epochs=50,
    eval_period=1,
    do_early_stop=False,
    criteria_key="eval_loss",
    seed=2024,
    save_dir="~/busy-beeway/transformers/logs",
    save_model=True,
):

    save_dir = osp.expanduser(save_dir)
    setup_logger(
        variant=None, seed=seed, base_log_dir=save_dir, include_exp_prefix_sub_dir=False
    )
    set_random_seed(seed)
    rng = np.random.default_rng(seed)
    data_size, query_len, observation_dim = training_data["observations"].shape
    eval_data_size = test_data["observations"].shape[0]
    interval = int(data_size / batch_size) + 1
    eval_interval = int(eval_data_size / batch_size) + 1
    imlp = MLP()
    model = InterventionMLPTrainer(
        imlp,
        observation_dim,
        decay_steps=int(n_epochs * interval),
        warmup_steps=int(n_epochs * interval * 0.1),
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
                        index_batch(training_data, shuffled_idx[start_pt:end_pt])
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
                    index_batch(test_data, list(range(eval_start_pt, eval_end_pt)))
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
