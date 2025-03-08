import os.path as osp

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import torch
from flax import nnx
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers.evaluation.eval_episodes import bb_run_episode
from transformers.models.dec_transformer import DT
from transformers.models.pref_transformer import PT
from transformers.training.jax_utils import batch_to_jax
from transformers.training.logging_utils import logger, setup_logger
from transformers.training.training import DecTransformerTrainer, PrefTransformerTrainer
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
    checkpointer = ocp.StandardCheckpointer()
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
    model_args = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_episode_steps": kwargs.get("max_episode_steps", int(max_episode_length)),
        "embd_dim": embd_dim,
        "pref_attn_embd_dim": kwargs.get("pref_attn_embd_dim", embd_dim),
        "num_heads": kwargs.get("num_heads", 4),
        "attn_dropout": kwargs.get("attn_dropout", 0.1),
        "resid_dropout": kwargs.get("resid_dropout", 0.1),
        "intermediate_dim": kwargs.get("intermediate_dim", 4 * embd_dim),
        "num_layers": kwargs.get("num_layers", 1),
        "embd_dropout": kwargs.get("embd_dropout", 0.1),
        "max_pos": kwargs.get("max_pos", max_pos),
        "eps": kwargs.get("eps", 0.1),
        "seed": seed,
    }
    trans = PT(
        state_dim=model_args["state_dim"],
        action_dim=model_args["action_dim"],
        max_episode_steps=model_args["max_episode_steps"],
        embd_dim=model_args["embd_dim"],
        pref_attn_embd_dim=model_args["pref_attn_embd_dim"],
        num_heads=model_args["num_heads"],
        attn_dropout=model_args["attn_dropout"],
        resid_dropout=model_args["resid_dropout"],
        intermediate_dim=model_args["intermediate_dim"],
        num_layers=model_args["num_layers"],
        embd_dropout=model_args["embd_dropout"],
        max_pos=model_args["max_pos"],
        eps=model_args["eps"],
        rngs=rngs,
    )
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
    checkpointer.wait_until_finished()
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
    eval_settings=[1, 10, 100, 500, 0],
    criteria_key="eval_loss",
    criteria_type="loss",
    save_dir="~/busy-beeway/transformers/logs",
    save=True,
    **kwargs,
):

    save_dir = osp.expanduser(save_dir)
    checkpointer = ocp.StandardCheckpointer()
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
    t_keys = jax.random.randint(rng_subkey, 1, 0, 10000)
    gen1 = torch.Generator().manual_seed(int(t_keys))

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
    model_args = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_episode_steps": kwargs.get("max_episode_steps", int(max_episode_length)),
        "embd_dim": embd_dim,
        "num_heads": kwargs.get("num_heads", 8),
        "attn_dropout": kwargs.get("attn_dropout", 0.1),
        "resid_dropout": kwargs.get("resid_dropout", 0.1),
        "intermediate_dim": kwargs.get("intermediate_dim", 4 * embd_dim),
        "num_layers": kwargs.get("num_layers", 6),
        "embd_dropout": kwargs.get("embd_dropout", 0.1),
        "max_pos": kwargs.get("max_pos", max_pos),
        "eps": kwargs.get("eps", 0.1),
        "seed": seed,
    }
    dec = DT(
        state_dim=model_args["state_dim"],
        action_dim=model_args["action_dim"],
        max_episode_steps=model_args["max_episode_steps"],
        embd_dim=model_args["embd_dim"],
        num_heads=model_args["num_heads"],
        attn_dropout=model_args["attn_dropout"],
        resid_dropout=model_args["resid_dropout"],
        intermediate_dim=model_args["intermediate_dim"],
        num_layers=model_args["num_layers"],
        embd_dropout=model_args["embd_dropout"],
        max_pos=model_args["max_pos"],
        eps=model_args["eps"],
        rngs=rngs,
    )
    # TODO: Control statement for different environments
    eval_sim = bb_run_episode
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
    if criteria_type == "acc":
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
                "eval_loss": [],
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
                    ep_return, ep_length = eval_sim(
                        model,
                        r_model,
                        move_stats,
                        rngs,
                        query_len,
                        eval_settings[2],
                        max_episode_length,
                    )
                    metrics["eval_loss"].append(
                        ((eval_settings[2] - ep_return) ** 2) / ep_length
                    )
                criteria = np.mean(metrics[criteria_key])

                if criteria_type == "acc":
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
                "eval_loss": [],
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
                    ep_return, ep_length = eval_sim(
                        model,
                        r_model,
                        move_stats,
                        rngs,
                        query_len,
                        eval_settings[2],
                        max_episode_length,
                    )
                    metrics["eval_loss"].append(
                        ((eval_settings[2] - ep_return) ** 2) / ep_length
                    )
                criteria = np.mean(metrics[criteria_key])
                if criteria_type == "acc":
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
    checkpointer.wait_until_finished()
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
