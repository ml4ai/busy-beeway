import os.path as osp

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import torch
from flax import nnx
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers.evaluation.eval_episodes import (
    bb_run_episode,
    run_antmaze_medium,
    bb_run_episode_IQL,
    run_antmaze_medium_IQL,
)
from transformers.models.dec_transformer import DT
from transformers.models.pref_transformer import PT
from transformers.models.policy import NormalTanhPolicy
from transformers.models.value_net import ValueCritic, DoubleCritic
from transformers.training.jax_utils import batch_to_jax
from transformers.training.logging_utils import logger, setup_logger
from transformers.training.training import (
    DecTransformerTrainer,
    PrefTransformerTrainer,
    IQLTrainer,
)
from transformers.training.utils import Timer, save_model


def train_pt(
    data,
    seed,
    train_split=0.7,
    batch_size=64,
    num_workers=2,
    n_epochs=50,
    eval_period=1,
    criteria_key="eval_loss",
    criteria_type="loss",
    save_dir="~/busy-beeway/transformers/logs",
    save=True,
    **kwargs,
):

    save_dir = osp.expanduser(save_dir)
    checkpointer = ocp.Checkpointer(ocp.CompositeCheckpointHandler())
    setup_logger(
        variant=None,
        seed=seed,
        base_log_dir=save_dir,
        include_exp_prefix_sub_dir=False,
    )

    state_shape, action_shape = data.shapes()
    _, query_len, state_dim = state_shape
    action_dim = action_shape[2]
    max_episode_length = data.max_episode_length()
    rng_key = jax.random.key(seed)
    rng_key, rng_subkey = jax.random.split(rng_key, 2)
    t_keys = jax.random.randint(rng_subkey, 2, 0, 10000)
    gen1 = torch.Generator().manual_seed(int(t_keys[0]))
    gen2 = torch.Generator().manual_seed(int(t_keys[1]))
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
    rng_subkey1, rng_subkey2, rng_subkey3 = jax.random.split(rng_key, 3)
    rngs = nnx.Rngs(rng_subkey1, params=rng_subkey2, dropout=rng_subkey3)
    max_pos = 2048
    while query_len > max_pos:
        max_pos *= 2
    embd_dim = kwargs.get("embd_dim", min(batch_size, 256))
    model_args = [
        state_dim,
        action_dim,
        kwargs.get("max_episode_steps", int(max_episode_length)),
        embd_dim,
        kwargs.get("pref_attn_embd_dim", embd_dim),
        kwargs.get("num_heads", 4),
        kwargs.get("attn_dropout", 0.1),
        kwargs.get("resid_dropout", 0.1),
        kwargs.get("intermediate_dim", 4 * embd_dim),
        kwargs.get("num_layers", 3),
        kwargs.get("embd_dropout", 0.1),
        kwargs.get("max_pos", max_pos),
        kwargs.get("eps", 0.1),
        seed,
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

    model = PrefTransformerTrainer(
        trans,
        init_value=kwargs.get("init_value", 0),
        peak_value=kwargs.get("peak_value", 1e-4),
        warmup_steps=kwargs.get("warmup_steps", int(n_epochs * interval * 0.1)),
        decay_steps=kwargs.get("decay_steps", int(n_epochs * interval)),
        end_value=kwargs.get("end_value", 0),
    )
    c_best_epoch = 0
    if criteria_type == "acc":
        c_criteria_key = -np.inf
    else:
        c_criteria_key = np.inf
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
            with Timer() as train_timer:
                for i, t_data in tqdm(
                    enumerate(training_data_loader),
                    total=interval,
                    desc=f"Training Epoch {epoch}",
                ):
                    batch = {}
                    (
                        batch["states"],
                        batch["actions"],
                        batch["timesteps"],
                        batch["attn_mask"],
                        batch["states_2"],
                        batch["actions_2"],
                        batch["timesteps_2"],
                        batch["attn_mask_2"],
                        batch["labels"],
                    ) = t_data
                    for k in batch:
                        batch[k] = jnp.asarray(batch[k])
                    batch = batch_to_jax(batch)
                    for key, val in model.train(batch).items():
                        metrics[key].append(val)
            metrics["train_time"] = train_timer()
        else:
            # for using early stopping with train loss.
            metrics["training_loss"] = np.nan

        # eval phase
        if epoch % eval_period == 0:
            for j, e_data in tqdm(
                enumerate(test_data_loader),
                total=eval_interval,
                desc=f"Evaluation Epoch {epoch}",
            ):
                batch = {}
                (
                    batch["states"],
                    batch["actions"],
                    batch["timesteps"],
                    batch["attn_mask"],
                    batch["states_2"],
                    batch["actions_2"],
                    batch["timesteps_2"],
                    batch["attn_mask_2"],
                    batch["labels"],
                ) = e_data
                for k in batch:
                    batch[k] = jnp.asarray(batch[k])
                batch = batch_to_jax(batch)
                for key, val in model.evaluation(batch).items():
                    metrics[key].append(val)
            criteria = np.mean(metrics[criteria_key])

            if criteria_type == "acc":
                if criteria >= c_criteria_key:
                    c_best_epoch = epoch
                    c_criteria_key = criteria
                    metrics["best_epoch"] = c_best_epoch
                    metrics[f"{criteria_key}_best"] = c_criteria_key
                    save_model(trans, model_args, "best_model", save_dir, checkpointer)
            else:
                if criteria <= c_criteria_key:
                    c_best_epoch = epoch
                    c_criteria_key = criteria
                    metrics["best_epoch"] = c_best_epoch
                    metrics[f"{criteria_key}_best"] = c_criteria_key
                    save_model(trans, model_args, "best_model", save_dir, checkpointer)
        for key, val in metrics.items():
            if isinstance(val, list):
                if len(val):
                    metrics[key] = np.mean(val)
                else:
                    metrics[key] = np.nan
        logger.record_dict(metrics)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)
    if save:
        save_model(trans, model_args, "model", save_dir, checkpointer)
    checkpointer.close()


def train_dt(
    data,
    r_model,
    move_stats,
    seed,
    output_type="A_F",
    batch_size=64,
    num_workers=2,
    n_epochs=50,
    eval_settings=[1, 10, 100, 0],
    criteria_key="eval_metric",
    criteria_type="max",
    save_dir="~/busy-beeway/transformers/logs",
    save=True,
    **kwargs,
):

    save_dir = osp.expanduser(save_dir)
    checkpointer = ocp.Checkpointer(ocp.CompositeCheckpointHandler())
    setup_logger(
        variant=None,
        seed=seed,
        base_log_dir=save_dir,
        include_exp_prefix_sub_dir=False,
    )

    state_shape, action_shape = data.shapes()
    _, query_len, state_dim = state_shape
    action_dim = action_shape[2]

    max_episode_length = data.max_episode_length()

    rng_key = jax.random.key(seed)
    rng_key, rng_subkey = jax.random.split(rng_key, 2)
    t_keys = jax.random.randint(rng_subkey, 2, 0, 10000)
    gen1 = torch.Generator().manual_seed(int(t_keys[0]))
    np_rng = np.random.default_rng(int(t_keys[1]))
    training_data_loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=gen1,
        pin_memory=True,
    )

    interval = len(training_data_loader)

    rng_subkey1, rng_subkey2, rng_subkey3 = jax.random.split(rng_key, 3)
    rngs = nnx.Rngs(rng_subkey1, params=rng_subkey2, dropout=rng_subkey3)
    max_pos = 2048
    while query_len > max_pos:
        max_pos *= 2
    embd_dim = kwargs.get("embd_dim", min(batch_size, 256))
    model_args = [
        state_dim,
        action_dim,
        kwargs.get("max_episode_steps", int(max_episode_length)),
        embd_dim,
        kwargs.get("num_heads", 2),
        kwargs.get("attn_dropout", 0.1),
        kwargs.get("resid_dropout", 0.1),
        kwargs.get("intermediate_dim", 4 * embd_dim),
        kwargs.get("num_layers", 3),
        kwargs.get("embd_dropout", 0.1),
        kwargs.get("max_pos", max_pos),
        kwargs.get("eps", 0.1),
        seed,
    ]
    dec = DT(
        state_dim=model_args[0],
        action_dim=model_args[1],
        max_episode_steps=model_args[2],
        embd_dim=model_args[3],
        num_heads=model_args[4],
        attn_dropout=model_args[5],
        resid_dropout=model_args[6],
        intermediate_dim=model_args[7],
        num_layers=model_args[8],
        embd_dropout=model_args[9],
        max_pos=model_args[10],
        eps=model_args[11],
        rngs=rngs,
    )
    if eval_settings[3] == 1:
        eval_sim = run_antmaze_medium
    else:
        eval_sim = bb_run_episode

    model_args = np.array(model_args)

    model = DecTransformerTrainer(
        dec,
        output_type,
        init_value=kwargs.get("init_value", 0),
        peak_value=kwargs.get("peak_value", 1e-4),
        warmup_steps=kwargs.get("warmup_steps", int(n_epochs * interval * 0.1)),
        decay_steps=kwargs.get("decay_steps", int(n_epochs * interval)),
        end_value=kwargs.get("end_value", 0),
    )
    c_best_epoch = 0
    if criteria_type == "max":
        c_criteria_key = -np.inf
    else:
        c_criteria_key = np.inf
    if output_type == "S_D" or output_type == "A_D":
        for epoch in range(n_epochs + 1):
            metrics = {
                "epoch": epoch,
                "train_time": np.nan,
                "training_loss": [],
                "training_acc": [],
                "eval_metric": [],
                "best_epoch": c_best_epoch,
                f"{criteria_key}_best": c_criteria_key,
            }
            if epoch:
                with Timer() as train_timer:
                    for i, t_data in tqdm(
                        enumerate(training_data_loader),
                        total=interval,
                        desc=f"Training Epoch {epoch}",
                    ):
                        batch = {}
                        (
                            batch["states"],
                            batch["actions"],
                            batch["timesteps"],
                            batch["attn_mask"],
                            batch["returns"],
                        ) = t_data
                        for k in batch:
                            batch[k] = jnp.asarray(batch[k])
                        batch = batch_to_jax(batch)
                        for key, val in model.train(batch).items():
                            metrics[key].append(val)
                metrics["train_time"] = train_timer()
            else:
                # for using early stopping with train loss.
                metrics["training_loss"] = np.nan

            # eval phase
            if epoch % eval_setting[0] == 0:
                for i in tqdm(
                    range(eval_settings[1]),
                    total=eval_settings[1],
                    desc=f"Evaluation Epoch {epoch}",
                ):
                    met = eval_sim(
                        dec,
                        r_model,
                        move_stats,
                        query_len,
                        eval_settings[2],
                        max_episode_length,
                        rng=np_rng,
                    )
                    metrics["eval_metric"].append(met)
                criteria = np.mean(metrics[criteria_key])

                if criteria_type == "max":
                    if criteria >= c_criteria_key:
                        c_best_epoch = epoch
                        c_criteria_key = criteria
                        metrics["best_epoch"] = c_best_epoch
                        metrics[f"{criteria_key}_best"] = c_criteria_key
                        save_model(
                            dec, model_args, "best_model", save_dir, checkpointer
                        )
                else:
                    if criteria <= c_criteria_key:
                        c_best_epoch = epoch
                        c_criteria_key = criteria
                        metrics["best_epoch"] = c_best_epoch
                        metrics[f"{criteria_key}_best"] = c_criteria_key
                        save_model(
                            dec, model_args, "best_model", save_dir, checkpointer
                        )

            for key, val in metrics.items():
                if isinstance(val, list):
                    if len(val):
                        metrics[key] = np.mean(val)
                    else:
                        metrics[key] = np.nan
            logger.record_dict(metrics)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)
        if save:
            save_model(dec, model_args, "model", save_dir, checkpointer)
    else:
        for epoch in range(n_epochs + 1):
            metrics = {
                "epoch": epoch,
                "train_time": np.nan,
                "training_loss": [],
                "eval_metric": [],
                "best_epoch": c_best_epoch,
                f"{criteria_key}_best": c_criteria_key,
            }
            if epoch:
                with Timer() as train_timer:
                    for i, t_data in tqdm(
                        enumerate(training_data_loader),
                        total=interval,
                        desc=f"Training Epoch {epoch}",
                    ):
                        batch = {}
                        (
                            batch["states"],
                            batch["actions"],
                            batch["timesteps"],
                            batch["attn_mask"],
                            batch["returns"],
                        ) = t_data
                        for k in batch:
                            batch[k] = jnp.asarray(batch[k])
                        batch = batch_to_jax(batch)
                        for key, val in model.train(batch).items():
                            metrics[key].append(val)
                metrics["train_time"] = train_timer()
            else:
                # for using early stopping with train loss.
                metrics["training_loss"] = np.nan

            # eval phase
            if epoch % eval_settings[0] == 0:
                for i in tqdm(
                    range(eval_settings[1]),
                    total=eval_settings[1],
                    desc=f"Evaluation Epoch {epoch}",
                ):
                    met = eval_sim(
                        dec,
                        r_model,
                        move_stats,
                        query_len,
                        eval_settings[2],
                        max_episode_length,
                        rng=np_rng,
                    )
                    metrics["eval_metric"].append(met)
                criteria = np.mean(metrics[criteria_key])
                if criteria_type == "max":
                    if criteria >= c_criteria_key:
                        c_best_epoch = epoch
                        c_criteria_key = criteria
                        metrics["best_epoch"] = c_best_epoch
                        metrics[f"{criteria_key}_best"] = c_criteria_key
                        save_model(
                            dec, model_args, "best_model", save_dir, checkpointer
                        )
                else:
                    if criteria <= c_criteria_key:
                        c_best_epoch = epoch
                        c_criteria_key = criteria
                        metrics["best_epoch"] = c_best_epoch
                        metrics[f"{criteria_key}_best"] = c_criteria_key
                        save_model(
                            dec, model_args, "best_model", save_dir, checkpointer
                        )

            for key, val in metrics.items():
                if isinstance(val, list):
                    if len(val):
                        metrics[key] = np.mean(val)
                    else:
                        metrics[key] = np.nan
            logger.record_dict(metrics)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)
        if save:
            save_model(dec, model_args, "model", save_dir, checkpointer)
    checkpointer.close()


def train_IQL(
    data,
    r_model,
    move_stats,
    seed,
    batch_size=64,
    num_workers=2,
    n_epochs=50,
    eval_settings=[1, 10, 100, 0],
    criteria_key="eval_metric",
    criteria_type="max",
    save_dir="~/busy-beeway/transformers/logs",
    save=True,
    **kwargs,
):

    save_dir = osp.expanduser(save_dir)
    checkpointer = ocp.Checkpointer(ocp.CompositeCheckpointHandler())
    setup_logger(
        variant=None,
        seed=seed,
        base_log_dir=save_dir,
        include_exp_prefix_sub_dir=False,
    )

    state_shape, action_shape = data.shapes()
    state_dim = state_shape[2]
    action_dim = action_shape[2]

    rng_key = jax.random.key(seed)
    rng_key, rng_subkey = jax.random.split(rng_key, 2)
    t_keys = jax.random.randint(rng_subkey, 2, 0, 10000)
    gen1 = torch.Generator().manual_seed(int(t_keys[0]))
    np_rng = np.random.default_rng(int(t_keys[1]))
    training_data_loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=gen1,
        pin_memory=True,
    )

    interval = len(training_data_loader)

    rng_subkey1, rng_subkey2, rng_subkey3, rng_subkey4 = jax.random.split(rng_key, 4)
    rngs = nnx.Rngs(rng_subkey1, params=rng_subkey2, dropout=rng_subkey3, sample=rng_subkey4)

    hidden_dims = (kwargs.get("hidden_dims", [256, 256]),)
    actor_args = [
        state_dim,
        hidden_dims[-1],
        hidden_dims[:-1],
        action_dim,
        kwargs.get("state_dependent_std", True),
        kwargs.get("dropout_rate", None),
        kwargs.get("log_std_scale", 1.0),
        kwargs.get("log_std_min", -10.0),
        kwargs.get("log_std_max", 2.0),
        kwargs.get("tanh_squash_distribution", True),
        seed,
    ]
    actor = NormalTanhPolicy(
        state_dim=actor_args[0],
        mlp_output_dim=actor_args[1],
        hidden_dims=actor_args[2],
        action_dim=actor_args[3],
        state_dependent_std=actor_args[4],
        dropout_rate=actor_args[5],
        log_std_scale=actor_args[6],
        log_std_min=actor_args[7],
        log_std_max=actor_args[8],
        tanh_squash_distribution=actor_args[9],
        rngs=rngs,
    )

    vCritic = ValueCritic(state_dim=state_dim, hidden_dims=hidden_dims)

    activations = kwargs.get("activations", nnx.relu)

    qCritic = DoubleCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        activations=activations,
    )

    tCritic = DoubleCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        activations=activations,
    )

    if eval_settings[3] == 1:
        eval_sim = run_antmaze_medium_IQL
    else:
        eval_sim = bb_run_episode_IQL

    actor_args = np.array(actor_args)

    trainer = IQLTrainer(
        actor,
        vCritic,
        qCritic,
        tCritic,
        opt_decay_schedule=kwargs.get("opt_decay_schedule", "cosine"),
        max_steps=kwargs.get("max_steps", int(1e6)),
        actor_lr=kwargs.get("actor_lr", 3e-4),
        value_lr=kwargs.get("value_lr", 3e-4),
        critic_lr=kwargs.get("critic_lr", 3e-4),
        expectile=kwargs.get("expectile", 0.8),
        temperature=kwargs.get("temperature", 0.1),
        discount=kwargs.get("discount", 0.99),
        tau=kwargs.get("tau", 0.005),
    )
    c_best_epoch = 0
    if criteria_type == "max":
        c_criteria_key = -np.inf
    else:
        c_criteria_key = np.inf
    for epoch in range(n_epochs + 1):
        metrics = {
            "epoch": epoch,
            "train_time": np.nan,
            "training_loss": [],
            "eval_metric": [],
            "best_epoch": c_best_epoch,
            f"{criteria_key}_best": c_criteria_key,
        }
        if epoch:
            with Timer() as train_timer:
                for i, t_data in tqdm(
                    enumerate(training_data_loader),
                    total=interval,
                    desc=f"Training Epoch {epoch}",
                ):
                    batch = {}
                    (
                        batch["states"],
                        batch["next_states"],
                        batch["actions"],
                        batch["timesteps"],
                        batch["attn_mask"],
                        batch["rewards"],
                    ) = t_data
                    for k in batch:
                        batch[k] = jnp.asarray(batch[k])
                    batch = batch_to_jax(batch)
                    for key, val in trainer.train(batch).items():
                        metrics[key].append(val)
            metrics["train_time"] = train_timer()
        else:
            # for using early stopping with train loss.
            metrics["training_loss"] = np.nan

        # eval phase
        if epoch % eval_settings[0] == 0:
            for i in tqdm(
                range(eval_settings[1]),
                total=eval_settings[1],
                desc=f"Evaluation Epoch {epoch}",
            ):
                met = eval_sim(
                    actor,
                    r_model,
                    move_stats,
                    max_episode_length,
                    rng=rngs,
                )
                metrics["eval_metric"].append(met)
            criteria = np.mean(metrics[criteria_key])
            if criteria_type == "max":
                if criteria >= c_criteria_key:
                    c_best_epoch = epoch
                    c_criteria_key = criteria
                    metrics["best_epoch"] = c_best_epoch
                    metrics[f"{criteria_key}_best"] = c_criteria_key
                    save_model(
                        actor, actor_args, "best_actor", save_dir, checkpointer
                    )
            else:
                if criteria <= c_criteria_key:
                    c_best_epoch = epoch
                    c_criteria_key = criteria
                    metrics["best_epoch"] = c_best_epoch
                    metrics[f"{criteria_key}_best"] = c_criteria_key
                    save_model(
                        actor, actor_args, "best_actor", save_dir, checkpointer
                    )

        for key, val in metrics.items():
            if isinstance(val, list):
                if len(val):
                    metrics[key] = np.mean(val)
                else:
                    metrics[key] = np.nan
        logger.record_dict(metrics)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)
    if save:
        save_model(actor, actor_args, "actor", save_dir, checkpointer)
    checkpointer.close()


# def train_mt(
#     data,
#     seed,
#     train_split=0.7,
#     batch_size=64,
#     num_workers=2,
#     n_epochs=50,
#     eval_period=1,
#     do_early_stop=False,
#     criteria_key="eval_loss",
#     save_dir="~/busy-beeway/transformers/logs",
#     save_model=True,
#     pretrained_params=None,
#     **kwargs,
# ):

#     save_dir = osp.expanduser(save_dir)
#     setup_logger(
#         variant=None,
#         seed=seed,
#         base_log_dir=save_dir,
#         include_exp_prefix_sub_dir=False,
#     )

#     state_shape, action_shape = data.shapes()
#     _, query_len, state_dim = state_shape
#     action_dim = action_shape[2]
#     if pretrained_params is None:
#         max_episode_length = data.max_episode_length()
#     else:
#         max_episode_length = (
#             pretrained_params["params"]["Embed_0"]["embedding"].shape[0] - 1
#         )
#     rng_key = jax.random.PRNGKey(seed)
#     rng_key, rng_subkey = jax.random.split(rng_key, 2)
#     gen1 = torch.Generator().manual_seed(int(rng_subkey[0]))
#     gen2 = torch.Generator().manual_seed(int(rng_subkey[1]))
#     training_data, test_data = random_split(
#         data, [train_split, 1 - train_split], generator=gen1
#     )
#     training_data_loader = DataLoader(
#         training_data,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#         generator=gen2,
#         pin_memory=True,
#     )
#     test_data_loader = DataLoader(
#         test_data,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         shuffle=False,
#         pin_memory=True,
#     )

#     interval = len(training_data_loader)
#     eval_interval = len(test_data_loader)
#     rng_subkey1, rng_subkey2, rng_subkey3 = jax.random.split(rng_key, 3)
#     max_pos = 2048
#     while query_len > max_pos:
#         max_pos *= 2
#     embd_dim = kwargs.get("embd_dim", min(batch_size, 256))
#     trans = MT(
#         state_dim=state_dim,
#         action_dim=action_dim,
#         max_episode_steps=kwargs.get("max_episode_steps", max_episode_length),
#         embd_dim=embd_dim,
#         pref_attn_embd_dim=kwargs.get("pref_attn_embd_dim", embd_dim),
#         num_heads=kwargs.get("num_heads", 4),
#         attn_dropout=kwargs.get("attn_dropout", 0.1),
#         resid_dropout=kwargs.get("resid_dropout", 0.1),
#         intermediate_dim=kwargs.get("intermediate_dim", 4 * embd_dim),
#         num_layers=kwargs.get("num_layers", 1),
#         embd_dropout=kwargs.get("embd_dropout", 0.1),
#         max_pos=kwargs.get("max_pos", max_pos),
#         eps=kwargs.get("eps", 0.1),
#     )
#     model = MentorTransformerTrainer(
#         trans,
#         rng_subkey1,
#         rng_subkey2,
#         pretrained_params,
#         init_value=kwargs.get("init_value", 0),
#         peak_value=kwargs.get("peak_value", 1e-4),
#         warmup_steps=kwargs.get("warmup_steps", int(n_epochs * interval * 0.1)),
#         decay_steps=kwargs.get("decay_steps", int(n_epochs * interval)),
#         end_value=kwargs.get("end_value", 0),
#     )
#     early_stop = EarlyStopping(min_delta=0, patience=0)
#     c_best_epoch = np.nan
#     c_criteria_key = np.nan
#     for epoch, (s_key, t_key, e_key) in enumerate(
#         jax.random.split(rng_subkey3, (n_epochs + 1, 3))
#     ):
#         metrics = {
#             "epoch": epoch,
#             "train_time": np.nan,
#             "training_loss": [],
#             "training_acc": [],
#             "eval_loss": [],
#             "eval_acc": [],
#             "best_epoch": c_best_epoch,
#             f"{criteria_key}_best": c_criteria_key,
#         }
#         if epoch:
#             with Timer() as train_timer:
#                 t_keys = jax.random.split(t_key, interval)
#                 for i, t_data in tqdm(
#                     enumerate(training_data_loader),
#                     total=interval,
#                     desc=f"Training Epoch {epoch}",
#                 ):
#                     batch = {}
#                     (
#                         batch["states"],
#                         batch["actions"],
#                         batch["timesteps"],
#                         batch["attn_mask"],
#                         batch["labels"],
#                     ) = t_data
#                     for k in batch:
#                         batch[k] = jnp.asarray(batch[k])
#                     batch = batch_to_jax(batch)
#                     for key, val in model.train(batch, t_keys[i]).items():
#                         metrics[key].append(val)
#             metrics["train_time"] = train_timer()
#         else:
#             # for using early stopping with train loss.
#             metrics["training_loss"] = np.nan

#         # eval phase
#         if epoch % eval_period == 0:
#             e_keys = jax.random.split(e_key, eval_interval)
#             for j, e_data in tqdm(
#                 enumerate(test_data_loader),
#                 total=eval_interval,
#                 desc=f"Evaluation Epoch {epoch}",
#             ):
#                 batch = {}
#                 (
#                     batch["states"],
#                     batch["actions"],
#                     batch["timesteps"],
#                     batch["attn_mask"],
#                     batch["labels"],
#                 ) = e_data
#                 for k in batch:
#                     batch[k] = jnp.asarray(batch[k])
#                 batch = batch_to_jax(batch)
#                 for key, val in model.evaluation(batch, e_keys[j]).items():
#                     metrics[key].append(val)
#             criteria = np.mean(metrics[criteria_key])
#             early_stop = early_stop.update(criteria)
#             if early_stop.should_stop and do_early_stop:
#                 for key, val in metrics.items():
#                     if isinstance(val, list):
#                         if len(val):
#                             metrics[key] = np.mean(val)
#                         else:
#                             metrics[key] = np.nan
#                 logger.record_dict(metrics)
#                 logger.dump_tabular(with_prefix=False, with_timestamp=False)
#                 print("Met early stopping criteria, breaking...")
#                 break
#             elif epoch > 0 and early_stop.has_improved:
#                 c_best_epoch = epoch
#                 c_criteria_key = criteria
#                 metrics["best_epoch"] = c_best_epoch
#                 metrics[f"{criteria_key}_best"] = c_criteria_key
#                 save_data = {"model": model, "epoch": epoch}
#                 save_pickle(save_data, "best_model.pkl", save_dir)

#         for key, val in metrics.items():
#             if isinstance(val, list):
#                 if len(val):
#                     metrics[key] = np.mean(val)
#                 else:
#                     metrics[key] = np.nan
#         logger.record_dict(metrics)
#         logger.dump_tabular(with_prefix=False, with_timestamp=False)
#     if save_model:
#         save_data = {"model": model, "epoch": epoch}
#         save_pickle(save_data, "model.pkl", save_dir)


# def train_mamlpt(
#     data,
#     seed,
#     train_val_test_split=(70, 5, 4),
#     N_way=5,
#     K_shot=4,
#     num_workers=2,
#     n_epochs=50,
#     eval_period=1,
#     do_early_stop=False,
#     criteria_key="eval_loss",
#     save_dir="~/busy-beeway/transformers/logs",
#     save_model=True,
#     **kwargs,
# ):

#     save_dir = osp.expanduser(save_dir)
#     setup_logger(
#         variant=None,
#         seed=seed,
#         base_log_dir=save_dir,
#         include_exp_prefix_sub_dir=False,
#     )

#     state_shape, action_shape = data.shapes()
#     _, query_len, state_dim = state_shape
#     action_dim = action_shape[2]
#     max_episode_length = data.max_episode_length()
#     rng_key = jax.random.PRNGKey(seed)
#     rng_key, rng_subkey = jax.random.split(rng_key, 2)
#     gen1 = torch.Generator().manual_seed(int(rng_subkey[0]))
#     gen2 = torch.Generator().manual_seed(int(rng_subkey[1]))
#     train_n, val_n, test_n = train_val_test_split
#     training, val, test = get_train_val_test_split(
#         data, train_n, val_n, test_n, gen=gen1
#     )
#     training_data, training_c_idx = training
#     validation_data, validation_c_idx = val
#     # The list of test candidates gets pickled, so that a test set can be built later.
#     _, test_c_idx = test

#     training_data_loader = DataLoader(
#         training_data,
#         batch_sampler=FewShotBatchSampler(
#             training_c_idx,
#             include_query=True,
#             N_way=N_way,
#             K_shot=K_shot,
#             shuffle=True,
#             gen=gen2,
#         ),
#         num_workers=num_workers,
#         pin_memory=True,
#     )
#     validation_data_loader = DataLoader(
#         validation_data,
#         batch_sampler=FewShotBatchSampler(
#             validation_c_idx,
#             include_query=True,
#             N_way=N_way,
#             K_shot=K_shot,
#             shuffle=False,
#         ),
#         num_workers=num_workers,
#         pin_memory=True,
#     )

#     interval = len(training_data_loader)
#     eval_interval = len(validation_data_loader)
#     max_pos = 2048
#     while query_len > max_pos:
#         max_pos *= 2
#     embd_dim = kwargs.get("embd_dim", 256)
#     trans = PT(
#         state_dim=state_dim,
#         action_dim=action_dim,
#         max_episode_steps=kwargs.get("max_episode_steps", max_episode_length),
#         embd_dim=embd_dim,
#         pref_attn_embd_dim=kwargs.get("pref_attn_embd_dim", embd_dim),
#         num_heads=kwargs.get("num_heads", 4),
#         attn_dropout=kwargs.get("attn_dropout", 0.1),
#         resid_dropout=kwargs.get("resid_dropout", 0.1),
#         intermediate_dim=kwargs.get("intermediate_dim", 4 * embd_dim),
#         num_layers=kwargs.get("num_layers", 1),
#         embd_dropout=kwargs.get("embd_dropout", 0.1),
#         max_pos=kwargs.get("max_pos", max_pos),
#         eps=kwargs.get("eps", 0.1),
#     )

#     rng_subkey1, rng_subkey2, rng_subkey3 = jax.random.split(rng_key, 3)
#     model = MAMLPTTrainer(
#         trans,
#         rng_subkey1,
#         rng_subkey2,
#         init_value=kwargs.get("init_value", 0),
#         peak_value=kwargs.get("peak_value", 1e-4),
#         warmup_steps=kwargs.get("warmup_steps", int(n_epochs * interval * 0.1)),
#         decay_steps=kwargs.get("decay_steps", int(n_epochs * interval)),
#         end_value=kwargs.get("end_value", 0),
#         inner_lr=kwargs.get("inner_lr", 0.01),
#     )
#     early_stop = EarlyStopping(min_delta=0, patience=0)
#     c_best_epoch = np.nan
#     c_criteria_key = np.nan
#     for epoch, (s_key, t_key, e_key) in enumerate(
#         jax.random.split(rng_subkey3, (n_epochs + 1, 3))
#     ):
#         metrics = {
#             "epoch": epoch,
#             "train_time": np.nan,
#             "training_loss": [],
#             "training_acc": [],
#             "eval_loss": [],
#             "eval_acc": [],
#             "best_epoch": c_best_epoch,
#             f"{criteria_key}_best": c_criteria_key,
#         }
#         if epoch:
#             with Timer() as train_timer:
#                 t_keys = jax.random.split(t_key, interval)
#                 for i, t_data in tqdm(
#                     enumerate(training_data_loader),
#                     total=interval,
#                     desc=f"Training Epoch {epoch}",
#                 ):
#                     batch = batch_to_jax(process_c_batch(t_data, N_way, K_shot))
#                     for key, val in model.train(batch, t_keys[i]).items():
#                         metrics[key].append(val)
#             metrics["train_time"] = train_timer()
#         else:
#             # for using early stopping with train loss.
#             metrics["training_loss"] = np.nan

#         # eval phase
#         if epoch % eval_period == 0:
#             e_keys = jax.random.split(e_key, eval_interval)
#             for j, e_data in tqdm(
#                 enumerate(validation_data_loader),
#                 total=eval_interval,
#                 desc=f"Evaluation Epoch {epoch}",
#             ):
#                 batch = batch_to_jax(process_c_batch(e_data, N_way, K_shot))
#                 for key, val in model.evaluation(batch, e_keys[j]).items():
#                     metrics[key].append(val)
#             criteria = np.mean(metrics[criteria_key])
#             early_stop = early_stop.update(criteria)
#             if early_stop.should_stop and do_early_stop:
#                 for key, val in metrics.items():
#                     if isinstance(val, list):
#                         if len(val):
#                             metrics[key] = np.mean(val)
#                         else:
#                             metrics[key] = np.nan
#                 logger.record_dict(metrics)
#                 logger.dump_tabular(with_prefix=False, with_timestamp=False)
#                 print("Met early stopping criteria, breaking...")
#                 break
#             elif epoch > 0 and early_stop.has_improved:
#                 c_best_epoch = epoch
#                 c_criteria_key = criteria
#                 metrics["best_epoch"] = c_best_epoch
#                 metrics[f"{criteria_key}_best"] = c_criteria_key
#                 save_data = {
#                     "model": model,
#                     "epoch": epoch,
#                     "training_participants": training_c_idx.keys(),
#                     "validation_participants": validation_c_idx.keys(),
#                     "test_participants": test_c_idx.keys(),
#                 }
#                 save_pickle(save_data, "best_model.pkl", save_dir)

#         for key, val in metrics.items():
#             if isinstance(val, list):
#                 if len(val):
#                     metrics[key] = np.mean(val)
#                 else:
#                     metrics[key] = np.nan
#         logger.record_dict(metrics)
#         logger.dump_tabular(with_prefix=False, with_timestamp=False)
#     if save_model:
#         save_data = {
#             "model": model,
#             "epoch": epoch,
#             "training_participants": training_c_idx.keys(),
#             "validation_participants": validation_c_idx.keys(),
#             "test_participants": test_c_idx.keys(),
#         }
#         save_pickle(save_data, "model.pkl", save_dir)

# def train_vt(
#     data,
#     seed,
#     train_split=0.7,
#     batch_size=64,
#     num_workers=2,
#     n_epochs=50,
#     eval_period=1,
#     do_early_stop=False,
#     criteria_key="eval_loss",
#     save_dir="~/busy-beeway/transformers/logs",
#     save_model=True,
#     pretrained_params=None,
#     **kwargs,
# ):

#     save_dir = osp.expanduser(save_dir)
#     setup_logger(
#         variant=None,
#         seed=seed,
#         base_log_dir=save_dir,
#         include_exp_prefix_sub_dir=False,
#     )

#     state_shape, _ = data.shapes()
#     _, query_len, state_dim = state_shape
#     if pretrained_params is None:
#         max_episode_length = data.max_episode_length()
#     else:
#         max_episode_length = (
#             pretrained_params["params"]["Embed_0"]["embedding"].shape[0] - 1
#         )
#     rng_key = jax.random.PRNGKey(seed)
#     rng_key, rng_subkey = jax.random.split(rng_key, 2)
#     gen1 = torch.Generator().manual_seed(int(rng_subkey[0]))
#     gen2 = torch.Generator().manual_seed(int(rng_subkey[1]))
#     training_data, test_data = random_split(
#         data, [train_split, 1 - train_split], generator=gen1
#     )
#     training_data_loader = DataLoader(
#         training_data,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#         generator=gen2,
#         pin_memory=True,
#     )
#     test_data_loader = DataLoader(
#         test_data,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         shuffle=False,
#         pin_memory=True,
#     )

#     interval = len(training_data_loader)
#     eval_interval = len(test_data_loader)
#     rng_subkey1, rng_subkey2, rng_subkey3 = jax.random.split(rng_key, 3)
#     max_pos = 512
#     while query_len > max_pos:
#         max_pos *= 2
#     embd_dim = kwargs.get("embd_dim", min(batch_size, 256))
#     val = VT(
#         state_dim=state_dim,
#         max_episode_steps=kwargs.get("max_episode_steps", max_episode_length),
#         embd_dim=embd_dim,
#         pref_attn_embd_dim=kwargs.get("pref_attn_embd_dim", embd_dim),
#         num_heads=kwargs.get("num_heads", 4),
#         attn_dropout=kwargs.get("attn_dropout", 0.1),
#         resid_dropout=kwargs.get("resid_dropout", 0.1),
#         intermediate_dim=kwargs.get("intermediate_dim", 4 * embd_dim),
#         num_layers=kwargs.get("num_layers", 1),
#         embd_dropout=kwargs.get("embd_dropout", 0.1),
#         max_pos=kwargs.get("max_pos", max_pos),
#         eps=kwargs.get("eps", 0.1),
#     )
#     model = ValTransformerTrainer(
#         val,
#         rng_subkey1,
#         rng_subkey2,
#         pretrained_params,
#         init_value=kwargs.get("init_value", 0),
#         peak_value=kwargs.get("peak_value", 1e-4),
#         warmup_steps=kwargs.get("warmup_steps", int(n_epochs * interval * 0.1)),
#         decay_steps=kwargs.get("decay_steps", int(n_epochs * interval)),
#         end_value=kwargs.get("end_value", 0),
#     )
#     early_stop = EarlyStopping(min_delta=0, patience=0)
#     c_best_epoch = np.nan
#     c_criteria_key = np.nan
#     for epoch, (s_key, t_key, e_key) in enumerate(
#         jax.random.split(rng_subkey3, (n_epochs + 1, 3))
#     ):
#         metrics = {
#             "epoch": epoch,
#             "train_time": np.nan,
#             "training_loss": [],
#             "eval_loss": [],
#             "best_epoch": c_best_epoch,
#             f"{criteria_key}_best": c_criteria_key,
#         }
#         if epoch:
#             with Timer() as train_timer:
#                 t_keys = jax.random.split(t_key, interval)
#                 for i, t_data in tqdm(
#                     enumerate(training_data_loader),
#                     total=interval,
#                     desc=f"Training Epoch {epoch}",
#                 ):
#                     batch = {}
#                     (
#                         batch["states"],
#                         batch["actions"],
#                         batch["timesteps"],
#                         batch["attn_mask"],
#                         batch["returns"],
#                     ) = t_data
#                     for k in batch:
#                         batch[k] = jnp.asarray(batch[k])
#                     batch = batch_to_jax(batch)
#                     for key, val in model.train(batch, t_keys[i]).items():
#                         metrics[key].append(val)
#             metrics["train_time"] = train_timer()
#         else:
#             # for using early stopping with train loss.
#             metrics["training_loss"] = np.nan

#         # eval phase
#         if epoch % eval_period == 0:
#             e_keys = jax.random.split(e_key, eval_interval)
#             for j, e_data in tqdm(
#                 enumerate(test_data_loader),
#                 total=eval_interval,
#                 desc=f"Evaluation Epoch {epoch}",
#             ):
#                 batch = {}
#                 (
#                     batch["states"],
#                     batch["actions"],
#                     batch["timesteps"],
#                     batch["attn_mask"],
#                     batch["returns"],
#                 ) = e_data
#                 for k in batch:
#                     batch[k] = jnp.asarray(batch[k])
#                 batch = batch_to_jax(batch)
#                 for key, val in model.evaluation(batch, e_keys[j]).items():
#                     metrics[key].append(val)
#             criteria = np.mean(metrics[criteria_key])
#             early_stop = early_stop.update(criteria)
#             if early_stop.should_stop and do_early_stop:
#                 for key, val in metrics.items():
#                     if isinstance(val, list):
#                         if len(val):
#                             metrics[key] = np.mean(val)
#                         else:
#                             metrics[key] = np.nan
#                 logger.record_dict(metrics)
#                 logger.dump_tabular(with_prefix=False, with_timestamp=False)
#                 print("Met early stopping criteria, breaking...")
#                 break
#             elif epoch > 0 and early_stop.has_improved:
#                 c_best_epoch = epoch
#                 c_criteria_key = criteria
#                 metrics["best_epoch"] = c_best_epoch
#                 metrics[f"{criteria_key}_best"] = c_criteria_key
#                 save_data = {"model": model, "epoch": epoch}
#                 save_pickle(save_data, "best_model.pkl", save_dir)

#         for key, val in metrics.items():
#             if isinstance(val, list):
#                 if len(val):
#                     metrics[key] = np.mean(val)
#                 else:
#                     metrics[key] = np.nan
#         logger.record_dict(metrics)
#         logger.dump_tabular(with_prefix=False, with_timestamp=False)
#     if save_model:
#         save_data = {"model": model, "epoch": epoch}
#         save_pickle(save_data, "model.pkl", save_dir)
